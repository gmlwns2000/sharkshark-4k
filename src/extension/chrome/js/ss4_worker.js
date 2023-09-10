chrome.runtime.onInstalled.addListener(function(details){
    if(details.reason == "install"){
        // 첫 설치시 실행할 코드
    }
    else if(details.reason == "update"){
        // 버전 업데이트 또는 확장 프로그램에서 새로고침시
    }
});

const proc_img = async (response, url) => {
    try {
        console.log('proc_img', url);

        let res = await fetch(url);
        let blob = await res.blob();

        let formData = new FormData();
        formData.append('file', blob);

        const server_url = "https://ainl.tk";
        const endpoint = server_url + "/upscale/image"
        console.log('proc_img POST', endpoint)
        res = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        console.log('proc_img response', url, res, blob);
        
        if(res.status == 200) {
            let json = await res.json();
            console.log('proc_img', json);
            if(json.url && json.result == 'ok') {
                let new_path;
                if(json.url[0] == '/') {
                    new_path = server_url + json.url
                } else {
                    throw "not implemented";
                }
                response({'status': 'ok', 'url': new_path, 'old_url': url, 'response': json});
            } else {
                response({'status': 'err', 'err': 'bad response', 'response': json});
            }
        } else {
            let json = 'bad text';
            try {
                json = res.responseText;
            } catch {}
            console.log(url, res, res.responseText);
            response({'status': 'err', 'err':'bad status', 'url': url, 'response': json});
        }
    } catch (err) {
        console.log('error ss4', err);
        response({'status': 'err', 'err': err})
    }
}

chrome.runtime.onMessage.addListener( (request,sender,response) => {
    try {
        switch (request.type) {
            case 'ss4_src':
                console.log('ss4_worker', request);
                proc_img(response, request.data);
                return true;
            default:
                response({'status': 'err', 'request': request, 'err': 'not handled'})
                break;
        }
    } catch (err) {
        console.log('error ss4', err);
        response({'status': 'err', 'request': request, 'err': err})
    }
});