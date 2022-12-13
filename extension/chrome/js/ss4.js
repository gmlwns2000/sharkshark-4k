/*jslint devel: true */
/*jslint es6 */
/*jslint browser: true */
'use strict';

function is_img_ok(img) {
    if (img.tagName != 'IMG') {
        return true;
    }

    if (!img.complete) {
        return false;
    }
    if (img.naturalWidth === 0) {
        return false;
    }
    return true;
}

function is_elem_visible(elem) {
    const paddingX = 0.5;
    const paddingY = 2.5;

    if(elem === '' || elem === undefined || elem === null) return false;
    // if (!(elem instanceof Element)) {
    //     console.log('elemvis', elem, typeof elem)
    //     return false;
    //     throw Error('is_elem_visible: elem is not an element.', elem);
    // }
    if(!(elem.getBoundingClientRect)) {
        console.log('elemvis', elem, typeof elem)
        return false;
    }
    const style = getComputedStyle(elem);
    if (style.display === 'none') return false;
    if (style.visibility !== 'visible') return false;
    if (style.opacity < 0.1) return false;
    if (elem.offsetWidth + elem.offsetHeight + elem.getBoundingClientRect().height +
        elem.getBoundingClientRect().width === 0) {
        return false;
    }
    
    const docuWidth = (document.documentElement.clientWidth || window.innerWidth);
    const docuHeight = (document.documentElement.clientHeight || window.innerHeight);
    const elemCenter = {
        x: elem.getBoundingClientRect().left + elem.offsetWidth / 2,
        y: elem.getBoundingClientRect().top + elem.offsetHeight / 2
    };
    // if (elemCenter.x < 0) return false;
    // if (elemCenter.x > docuWidth) return false;
    // if (elemCenter.y < 0) return false;
    // if (elemCenter.y > docuHeight) return false;
    const checkPos = (x, y) => {
        if( x < -docuWidth*paddingX || x > docuWidth*(1+paddingX) ||
            y < -docuHeight*paddingY || y > docuHeight*(1+paddingY)) {
            // console.log(x, y, docuWidth, docuHeight, padding, elem);
            return false;
        }
        return true
    }
    if(!(
        checkPos(elemCenter.x, elemCenter.y) ||
        checkPos(elem.getBoundingClientRect().left, elem.getBoundingClientRect().top) ||
        checkPos(elem.getBoundingClientRect().left, elem.getBoundingClientRect().bottom) ||
        checkPos(elem.getBoundingClientRect().right, elem.getBoundingClientRect().top) ||
        checkPos(elem.getBoundingClientRect().right, elem.getBoundingClientRect().bottom))
    ) return false;
    // let pointContainer = document.elementFromPoint(elemCenter.x, elemCenter.y);
    // do {
    //     if (pointContainer === elem) return true;
    // } while (pointContainer = pointContainer.parentNode);
    return true;
}

function url_ban(url) {
    if(url === null || url === undefined) { // if null
        return true;
    }

    if (typeof url === 'string' || url instanceof String) {
        if(url.includes('.svg')) {
            return true;
        }
        if(url.includes('svg+xml')) {
            return true;
        }
        if(url.includes('ainl.tk')) {
            //remove from upscaling server
            return true;
        }
    }
    return false;
}

function parse_css_url(css_url) {
    if(!css_url) return;

    let regex = css_url.match(/^\s*url\(\s*(.*)\s*\)\s*$/);
    if(!regex) return;

    let uri = regex[1];
    if(uri && uri.length && uri.slice) {
        let last = uri.length - 1;
        if (uri[0] === '"' && uri[last] === '"' || uri[0] === "'" && uri[last] === "'") {
            uri = uri.slice(1, -1);
        }
    }
    return uri;
}

async function proc_img(img, event) {
    // console.log('proc_img', img.src)
    let source_url_method;
    let source_url;
    if(img.tagName == 'IMG') {
        source_url_method = 'img';
        source_url = img.src;
        // console.log('imghelp', img)
    } else if (img.tagName == 'DIV') {
        if(img.style['backgroundImage'] && parse_css_url(img.style['backgroundImage'])) {
            source_url_method = 'css_background_image'
            source_url = parse_css_url(img.style['backgroundImage'])
            // console.log('proc_img css', source_url, img)
        }
    }

    if(
        (!img.ss4_status || img.ss4_status === 'pending' || (img.old_src && (img.ss4_src !== img.src))) &&
        (!url_ban(source_url)) &&
        source_url && is_elem_visible(img) &&
        ((img.naturalWidth / (img.clientWidth + 0.00001)) < 4.0 || source_url_method !== 'img')
    ) {
        img.ss4_status = 'working'
        
        try {
            let promise = new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({
                    type:'ss4_src', 
                    data:source_url
                }, (response) => {
                    resolve(response);
                });
            });
            let response = await promise;
            // console.log('proc_img worker response', response);
            
            if(response && response.status && response.status === 'ok' && response.url) {
                let new_url = response.url;
                console.log('proc_img updated url', new_url, response,
                // (!img.ss4_status || img.ss4_status === 'pending' || (img.old_src && (img.ss4_src !== img.src))),
                // (!url_ban(source_url)),
                // source_url, is_elem_visible(img),
                // ((img.naturalWidth / (img.clientWidth + 0.00001)) < 4.0 || source_url_method !== 'img'),
                // img.src
                );
                let patched = false;
                switch(source_url_method) {
                    case "img":
                        if(img.src == source_url) {
                            img.src = new_url;
                            patched = true;
                        }
                        break;
                    case "css_background_image":
                        img.style['backgroundImage'] = `url("${new_url}")`
                        // console.log('css update', img.style['backgroundSize'])
                        if( img.style['backgroundSize'] === null ||
                            img.style['backgroundSize'] === undefined ||
                            img.style['backgroundSize'] == ''
                        ) {
                            img.style['backgroundSize'] = '100%';
                        }
                        patched = true;
                        break;
                }
                if(patched) {
                    if(img.srcset) {
                        img.removeAttribute('srcset')
                    }
                    img.old_src = source_url;
                    img.ss4_src = new_url;
                    img.onerror = undefined;
                    img.removeAttribute('onerror')
                    if(!img.width || img.width < 4000) {
                        img.width = img.clientWidth;
                        img.height = img.clientHeight;
                    }
                    img.ss4_status = 'done';
                } else {
                    console.log('patched failed because src updated')
                    img.ss4_status = 'pending';
                }
            } else {
                console.log('proc_img failed url', source_url, response);
                img.ss4_status = 'pending';
            }
        } catch (err) {
            img.ss4_status = 'pending';
            throw err;
        }
    }
};

let proc_all_last = 0;
async function proc_all() {
    if((Date.now() - proc_all_last) < 100) {
        return;
    }
    proc_all_last = Date.now();
    console.log('iter');

    const patch = (elems) => {
        elems.forEach((elem, i) => {
            // console.log('pre', img.src, !img.ss4_inject, img.ss4_inject == 'pending', img.ss4_inject)
            if(!elem.ss4_inject || elem.ss4_inject == 'pending') {
                // img.crossOrigin = '';
                elem.addEventListener('load', (event) => {
                    proc_img(elem, event);
                })
                elem.ss4_inject = true;
            }
    
            if(is_img_ok(elem)) {
                proc_img(elem, null);
            }
        });
    }

    const patch_document = (docu) => {
        if(!docu.querySelectorAll) return;

        let imgs = docu.querySelectorAll("img");
        patch(imgs);
        let divs = docu.querySelectorAll("div");
        patch(divs);

        try {
            let iframes = docu.querySelectorAll("iframe");
            iframes.forEach((item, i) => {
                let idocument = item?.contentWindow?.document;
                if(idocument) {
                    patch_document(idocument);
                }
            })
        }
        catch (err) {
            // console.log('patch iframe failed because of error, perhaps CORS')
        }
    }
    patch_document(document);
};

let initialied = false;
if(!initialied) {
    initialied = true;
    proc_all();
    setInterval(proc_all, 5000);
    window.addEventListener('scroll', proc_all);
    window.addEventListener("click", () => setTimeout(proc_all, 150));
}