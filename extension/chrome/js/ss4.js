/*jslint devel: true */
/*jslint es6 */
/*jslint browser: true */

function toFit(
    cb,
    { dismissCondition = () => false, triggerCondition = () => true }
  ) {
    if (!cb) {
        throw Error('Invalid required arguments')
    }

    let tick = false

    return function() {
        console.log('scroll call')

        if (tick) {
            return
        }

        tick = true
        return requestAnimationFrame(() => {
            if (dismissCondition()) {
                tick = false
                return
            }

            if (triggerCondition()) {
                console.log('real call')
                tick = false
                return cb()
            }
        })
    }
}

function is_img_ok(img) {
    if (!img.complete) {
        return false;
    }
    if (img.naturalWidth === 0) {
        return false;
    }
    return true;
};

function is_elem_visible(elem) {
    const paddingX = 0.5;
    const paddingY = 2.5;
    if (!(elem instanceof Element)) throw Error('DomUtil: elem is not an element.');
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
    if (typeof url === 'string' || url instanceof String) {
        if(url.includes('.svg')) {
            // console.log('url ban', url)
            return true;
        }
        if(url.includes('svg+xml')) {
            // console.log('url ban', url)
            return true;
        }
    }
    return false;
}

async function proc_img(img, event) {
    // console.log('proc_img', img.src)
    if(
        (!img.ss4_status || img.ss4_status === 'pending' || (img.old_src && (img.ss4_src !== img.src))) && 
        img.src && is_elem_visible(img) &&
        ((img.naturalWidth / (img.clientWidth + 0.00001)) < 4.0) &&
        (!url_ban(img.src))
    ) {
        img.ss4_status = 'working'
        // console.log(img.src)
        
        try {
            let promise = new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({
                    type:'ss4_src', 
                    data:img.src
                }, (response) => {
                    resolve(response);
                });
            });
            let response = await promise;
            // console.log('proc_img worker response', response);
            
            if(response && response.status && response.status === 'ok' && response.url) {
                let new_url = response.url;
                console.log('proc_img updated url', new_url, response);
                img.old_src = img.src;
                img.ss4_src = new_url;
                img.width = img.clientWidth;
                img.height = img.clientHeight;
                img.src = new_url;
                img.ss4_status = 'done';
            } else {
                console.log('proc_img failed url', img.src, response);
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
    let imgs = document.querySelectorAll("img");
    imgs.forEach((img, i) => {
        // console.log('pre', img.src, !img.ss4_inject, img.ss4_inject == 'pending', img.ss4_inject)
        if(!img.ss4_inject || img.ss4_inject == 'pending') {
            // img.crossOrigin = '';
            img.addEventListener('load', (event) => {
                proc_img(img, event);
            })
            img.ss4_inject = true;
        }

        if(is_img_ok(img)) {
            proc_img(img, null);
        }
    });
};

let initialied = false;
if(!initialied) {
    initialied = true;
    proc_all();
    setInterval(proc_all, 5000);
    window.addEventListener('scroll', proc_all);
}