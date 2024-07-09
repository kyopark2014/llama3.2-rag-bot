const langstate = 'korean'; // korean or english
// earn endpoint 
let endpoint = localStorage.getItem('wss_url');
if(endpoint=="") {
    console.log('provisioning is required!');
}
console.log('endpoint: ', endpoint);

console.log('feedback...');
const feedback = document.getElementById('feedback');
feedback.style.display = 'none';    

let webSocket
let isConnected = false;
webSocket = connect(endpoint, 'initial');

// Documents
const title = document.querySelector('#title');
const sendBtn = document.querySelector('#sendBtn');
const message = document.querySelector('#chatInput')
const chatPanel = document.querySelector('#chatPanel');

HashMap = function() {
    this.map = new Array();
};

HashMap.prototype = {
    put: function(key, value) {
        this.map[key] = value;
    },
    get: function(key) {
        return this.map[key];
    },
    size: function() {
        var keys = new Array();
        for(i in this.map) {
            keys.push(i);
        }
        return keys.length;
    },
    remove: function(key) {
        delete this.map[key];
    },
    getKeys: function() {
        var keys = new Array();
        for(i in this.map) {
            keys.push(i);
        }
        return keys;
    }
};

let isResponsed = new HashMap();
let indexList = new HashMap();
let retryNum = new HashMap();

// message log list
let msglist = [];
let maxMsgItems = 200;
let msgHistory = new HashMap();
let sentTime = new HashMap();

let undelivered = new HashMap();
let retry_count = 0;
function sendMessage(message) {
    if(!isConnected) {
        console.log('reconnect...'); 
        webSocket = connect(endpoint, 'reconnect');
        
        if(langstate=='korean') {
            addNotifyMessage("재연결중입니다. 연결후 자동 재전송합니다.");
        }
        else {
            addNotifyMessage("We are connecting again. Your message will be retried after connection.");                        
        }

        undelivered.put(message.request_id, message);
        console.log('undelivered message: ', message);
        
        return false
    }
    else {
        webSocket.send(JSON.stringify(message));     
        console.log('message: ', message);   

        return true;
    }     
}

let tm;
let chromeTimer = 300;
function ping() {
    console.log('->ping');
    webSocket.send('__ping__');

    /* chromeTimer = chromeTimer - 50;  
    if(chromeTimer<60) {  // in order prevent session close by the chrome timer, 5min.
        chromeTimer = 300;
        window.location.href = "chat.html";  // refresh browser per 4min
    } */

    tm = setTimeout(function () {
        console.log('reconnect...');    
        
        webSocket = connect(endpoint, 'reconnect');
    }, 5000);
}
function pong() {
    clearTimeout(tm);
}

function connect(endpoint, type) {
    const ws = new WebSocket(endpoint);

    // connection event
    ws.onopen = function () {
        console.log('connected...');
        isConnected = true;

        if(undelivered.size() && retry_count>0) {
            let keys = undelivered.getKeys();
            console.log('retry undelived messags!');            
            console.log('keys: ', keys);
            console.log('retry_count: ', retry_count);

            for(i in keys) {
                let message = undelivered.get(keys[i])
                console.log('message', message)
                if(!sendMessage(message)) break;
                else {
                    undelivered.remove(message.request_id)
                }
            }
            retry_count--;
        }
        else {
            retry_count = 3
        }

        if(type == 'initial')
            setInterval(ping, 40000);  // ping interval: 40 seconds
    };

    // message 
    ws.onmessage = function (event) {        
        if (event.data.substr(1,8) == "__pong__") {
            console.log('<-pong');
            pong();
            return;
        }
        else {
            response = JSON.parse(event.data)

            if(response.request_id) {
                if(!indexList.get(response.request_id+':receive')) { // the first received message
                    let current = new Date();
                    let elapsed = (current - sentTime.get(response.request_id))/1000;
                    // console.log('elapsed time: ', elapsed);
                }
                // console.log('response: ', response);

                if(response.status == 'completed') {          
                    feedback.style.display = 'none';          
                    console.log('received message: ', response.msg);                  
                    addReceivedMessage(response.request_id, response.msg);  
                }                
                else if(response.status == 'istyping') {
                    feedback.style.display = 'inline';
                    // feedback.innerHTML = '<i>typing a message...</i>'; 
                }
                else if(response.status == 'proceeding') {
                    feedback.style.display = 'none';
                    addReceivedMessage(response.request_id, response.msg);  
                }                
                else if(response.status == 'debug') {
                    feedback.style.display = 'none';
                    console.log('debug: ', response.msg);
                    // addNotifyMessage(response.msg);
                    addReceivedMessage(response.request_id, response.msg);  
                }          
                else if(response.status == 'error') {
                    feedback.style.display = 'none';
                    console.log('error: ', response.msg);

                    if(response.msg.indexOf('throttlingException') || response.msg.indexOf('Too many requests') || response.msg.indexOf('too many requests')) {
                        addNotifyMessage('허용된 요청수를 초과하였습니다. 추후 다시 재도시도 해주세요.');  
                    }
                    else {
                        addNotifyMessage(response.msg);
                    }
                }   
            }
            else {
                console.log('system message: ', event.data);
            }
        }        
    };

    // disconnect
    ws.onclose = function () {
        console.log('disconnected...!');
        isConnected = false;

        ws.close();
        console.log('the session will be closed');
    };

    // error
    ws.onerror = function (error) {
        console.log(error);

        ws.close();
        console.log('the session will be closed');
    };

    return ws;
}

let callee = "AWS";
let index=0;

let userId = localStorage.getItem('userId'); // set userID if exists 
if(userId=="") {
    userId = uuidv4();
}
console.log('userId: ', userId);

let conversationType = localStorage.getItem('conv_type'); // set conv_type if exists 
if(conversationType=="") {
    conversationType = "normal";
}
console.log('conversationType: ', conversationType);

for (i=0;i<maxMsgItems;i++) {
    msglist.push(document.getElementById('msgLog'+i));

    // add listener        
    (function(index) {
        msglist[index].addEventListener("click", function() {
            if(msglist.length < maxMsgItems) i = index;
            else i = index + maxMsgItems;

            console.log('click! index: '+index);
        })
    })(i);
}

calleeName.textContent = "Chatbot";  
calleeId.textContent = "AWS";

if(langstate=='korean') {
    addNotifyMessage("Amazon Bedrock을 이용하여 채팅을 시작합니다.");
    addReceivedMessage(uuidv4(), "Amazon Bedrock을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다.")
}
else {
    addNotifyMessage("Start chat with Amazon Bedrock");             
    addReceivedMessage(uuidv4(), "Welcome to Amazon Bedrock. Use the conversational chatbot and summarize documents, TXT, PDF, and CSV. ")           
}

// get history
function getAllowTime() {    
    let allowableDays = 2; // two day's history
    
    let current = new Date();
    let allowable = new Date(current.getTime() - 24*60*60*1000*allowableDays);  
    let allowTime = getDate(allowable)+' '+getTime(current);
    console.log('Current Time: ', getDate(current)+' '+getTime(current));
    console.log('Allow Time: ', allowTime);
    
    return allowTime;
}
let allowTime = getAllowTime();
getHistory(userId, allowTime);

// Listeners
message.addEventListener('keyup', function(e){
    if (e.keyCode == 13) {
        onSend(e);
    }
});

// refresh button
refreshChatWindow.addEventListener('click', function(){
    console.log('go back user input menu');
    window.location.href = "index.html";
});

// depart button
depart.addEventListener('click', function(){
    console.log('depart icon');
    
    deleteItems(userId);    
});

sendBtn.addEventListener('click', onSend);
function onSend(e) {
    e.preventDefault();
    
    if(message.value != '') {
        console.log("msg: ", message.value);

        let current = new Date();
        let datastr = getDate(current);
        let timestr = getTime(current);
        let requestTime = datastr+' '+timestr

        let requestId = uuidv4();
        addSentMessage(requestId, timestr, message.value);
        
        if(conversationType=='qa-opensearch-vector') {
            type = "text",
            conv_type = 'qa-opensearch-vector',
            rag_type = 'opensearch',
            function_type = 'rag'
        }
        else if(conversationType=='qa-opensearch-hybrid') {
            type = "text",
            conv_type = 'qa-opensearch-hybrid',
            rag_type = 'opensearch',
            function_type = 'rag'
        }
        else {
            type = "text",
            conv_type = conversationType,
            rag_type = ''
            function_type = ''
        }
        
        sendMessage({
            "user_id": userId,
            "request_id": requestId,
            "request_time": requestTime,        
            "type": type,
            "body": message.value,
            "conv_type": conv_type,
            "rag_type": rag_type,
            "function_type": function_type
        })
        
        sentTime.put(requestId, current);
    }
    message.value = "";

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

function uuidv4() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

(function() {
    window.addEventListener("focus", function() {
//        console.log("Back to front");

        // if(msgHistory.get(callee))
        //    updateCallLogToDisplayed();
    })
})();

function getDate(current) {    
    return current.toISOString().slice(0,10);
}

function getTime(current) {
    let time_map = [current.getHours(), current.getMinutes(), current.getSeconds()].map((a)=>(a < 10 ? '0' + a : a));
    return time_map.join(':');
}

function addSentMessage(requestId, timestr, text) {
    let idx = index;

    if(!indexList.get(requestId+':send')) {
        indexList.put(requestId+':send', idx);

        index++;
    }
    else {
        idx = indexList.get(requestId+':send');
        // console.log("reused index="+idx+', id='+requestId+':send');        
    }
    // console.log("index (sendMessage):", idx);   

    let length = text.length;    
    // console.log('length: ', length);
    if(length < 10) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender20 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;   
    }
    else if(length < 14) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender25 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;   
    }
    else if(length < 17) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender30 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }  
    else if(length < 21) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender35 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }
    else if(length < 26) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender40 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }
    else if(length < 35) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender50 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }
    else if(length < 80) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender60 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }  
    else if(length < 145) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender70 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }  
    else {
        msglist[idx].innerHTML = 
            `<div class="chat-sender80 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }     

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}       

function addSentMessageForSummary(requestId, timestr, text) {  
    console.log("sent message: "+text);

    if(!indexList.get(requestId+':send')) {
        indexList.put(requestId+':send', index);             
    }
    else {
        index = indexList.get(requestId+':send');
        console.log("reused index="+index+', id='+requestId+':send');        
    }
    console.log("index:", index);   

    let length = text.length;
    if(length < 100) {
        msglist[index].innerHTML = 
            `<div class="chat-sender60 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;   
    }
    else {
        msglist[index].innerHTML = 
            `<div class="chat-sender80 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${index}"></h2></div>`;
    }   

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
    index++;
}  

function addReceivedMessage(requestId, msg) {
    // console.log("add received message: "+msg);
    sender = "Chatbot"

    let idx = index;

    // console.log("add received message: "+msg);
    if(!indexList.get(requestId+':receive')) {
        indexList.put(requestId+':receive', idx);
        index++;
    }
    else {
        idx = indexList.get(requestId+':receive');
        // console.log("reused index="+idx+', id='+requestId+':receive');        
    }
    // console.log("index (receiveMessage):", idx);   

    msg = msg.replaceAll("\n", "<br/>");

    var length = msg.length;
    // console.log('msg: ', msg)
    // console.log("length: ", length);

    if(length < 10) {
        msglist[idx].innerHTML = `<div class="chat-receiver20 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 14) {
        msglist[idx].innerHTML = `<div class="chat-receiver25 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 17) {
        msglist[idx].innerHTML = `<div class="chat-receiver30 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 21) {
        msglist[idx].innerHTML = `<div class="chat-receiver35 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 25) {
        msglist[idx].innerHTML = `<div class="chat-receiver40 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 35) {
        msglist[idx].innerHTML = `<div class="chat-receiver50 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 80) {
        msglist[idx].innerHTML = `<div class="chat-receiver60 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 145) {
        msglist[idx].innerHTML = `<div class="chat-receiver70 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else {
        msglist[idx].innerHTML = `<div class="chat-receiver80 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom    
}

function addNotifyMessage(msg) {
    console.log("index:", index);   

    msglist[index].innerHTML =  
        `<div class="notification-text">${msg}</div>`;     

    index++;

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

function updateChatHistory() {
    /* for(let i=0;i<maxMsgItems;i++) {
        msglist[i].innerHTML = `<div></div>`
    }
    
    msglist = [];
    index = 0;

    indexList = new HashMap();

    for (i=0;i<maxMsgItems;i++) {
        msglist.push(document.getElementById('msgLog'+i));
    
        // add listener        
        (function(index) {
            msglist[index].addEventListener("click", function() {
                if(msglist.length < maxMsgItems) i = index;
                else i = index + maxMsgItems;
    
                console.log('click! index: '+index);
            })
        })(i);
    } 

    getHistory(userId, 'update');    */ 
    window.location.href = "chat.html";
}

refreshChatWindow.addEventListener('click', function(){
    console.log('update chat window');
    // updateChatWindow(callee);
});

attachFile.addEventListener('click', function(){
    console.log('click: attachFile');

    let input = $(document.createElement('input')); 
    input.attr("type", "file");
    input.trigger('click');    

    $(document).ready(function() {
        input.change(function(evt) {
            var input = this;
            var url_file = $(this).val();
            var ext = url_file.substring(url_file.lastIndexOf('.') + 1).toLowerCase();
            //var filename = url_file.substring(url_file.lastIndexOf('\\') + 1).toLowerCase();
            var filename = url_file.substring(url_file.lastIndexOf('\\') + 1);

            console.log('url: ' + url_file);
            console.log('filename: ' + filename);
            console.log('ext: ' + ext);

            if(ext == 'pdf') {
                contentType = 'application/pdf'           
            }
            else if(ext == 'txt') {
                contentType = 'text/plain'
            }
            else if(ext == 'csv') {
                contentType = 'text/csv'
            }
            else if(ext == 'ppt') {
                contentType = 'application/vnd.ms-powerpoint'
            }
            else if(ext == 'pptx') {
                contentType = 'application/vnd.ms-powerpoint'
            }
            else if(ext == 'doc' || ext == 'docx') {
                contentType = 'application/msword'
            }
            else if(ext == 'xls') {
                contentType = 'application/vnd.ms-excel'
            }
            else if(ext == 'py') {
                contentType = 'application/x-python-code'
            }
            else if(ext == 'js') {
                contentType = 'application/javascript'
            }
            else if(ext == 'md') {
                contentType = 'text/markdown'
            }
            else if(ext == 'png') {
                contentType = 'image/png'
            }
            else if(ext == 'jpeg' || ext == 'jpg') {
                contentType = 'image/jpeg'
            }
            console.log('contentType: ', contentType)

            let current = new Date();
            let datastr = getDate(current);
            let timestr = getTime(current);
            let requestTime = datastr+' '+timestr
            let requestId = uuidv4();

            let command = message.value;
            if(ext == 'png' || ext == 'jpeg' || ext == 'jpg') {
                addSentMessageForSummary(requestId, timestr, message.value+"<br>"+"uploading the selected file in order to summarize...");

                message.value = "";
            }
            else {
                addSentMessageForSummary(requestId, timestr, "uploading the selected file in order to summarize...");
            }
            
            const uri = "upload";
            const xhr = new XMLHttpRequest();
        
            xhr.open("POST", uri, true);
            xhr.onreadystatechange = () => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    response = JSON.parse(xhr.responseText);
                    console.log("response: " + JSON.stringify(response));
                                        
                    // upload the file
                    const body = JSON.parse(response.body);
                    console.log('body: ', body);

                    const uploadURL = body.UploadURL;                    
                    console.log("UploadURL: ", uploadURL);

                    var xmlHttp = new XMLHttpRequest();
                    xmlHttp.open("PUT", uploadURL, true);       

                    //let formData = new FormData();
                    //formData.append("attachFile" , input.files[0]);
                    //console.log('uploading file info: ', formData.get("attachFile"));

                    const blob = new Blob([input.files[0]], { type: contentType });

                    xmlHttp.onreadystatechange = function() {
                        if (xmlHttp.readyState == XMLHttpRequest.DONE && xmlHttp.status == 200 ) {
                            console.log(xmlHttp.responseText);

                            function_type = 'upload'
                            if(conversationType=='qa-all') {
                                conv_type = 'qa',
                                rag_type = 'all'
                            }
                            else if(conversationType=='qa-opensearch-os') {
                                conv_type = 'qa-opensearch-vector',
                                rag_type = 'opensearch'
                            }
                            else if(conversationType=='qa-opensearch-hybrid') {
                                conv_type = 'qa-opensearch-hybrid',
                                rag_type = 'opensearch'
                            }
                            else if(conversationType=='qa-faiss') {
                                conv_type = 'qa',
                                rag_type = 'faiss'
                            }
                            else {
                                conv_type = conversationType,
                                rag_type = ''
                            }

                            // summary for the upload file                            
                            sendMessage({
                                "user_id": userId,
                                "request_id": requestId,
                                "request_time": requestTime,
                                "type": "document",
                                "body": filename,
                                "command": command,
                                "conv_type": conv_type,
                                "rag_type": rag_type,
                                "function_type": function_type
                            })
                        }
                        else if(xmlHttp.readyState == XMLHttpRequest.DONE && xmlHttp.status != 200) {
                            console.log('status' + xmlHttp.status);
                            alert("Try again! The request was failed.");
                        }
                    };
        
                    xmlHttp.send(blob); 
                    // xmlHttp.send(formData); 
                    console.log(xmlHttp.responseText);
                }
            };
        
            var requestObj = {
                "filename": filename,
                "contentType": contentType,
            }
            console.log("request: " + JSON.stringify(requestObj));
        
            var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});
        
            xhr.send(blob);       
        });
    });
       
    return false;
});

function getHistory(userId, allowTime) {
    const uri = "history";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            let history = JSON.parse(response['msg']);
            console.log("history: " + JSON.stringify(history));
                        
            for(let i=0; i<history.length; i++) {
                if(history[i].type=='text') {                
                    // let timestr = history[i].request_time.substring(11, 19);
                    let requestId = history[i].request_id;
                    console.log("requestId: ", requestId);
                    let timestr = history[i].request_time;
                    console.log("timestr: ", timestr);
                    let body = history[i].body;
                    console.log("question: ", body);
                    let msg = history[i].msg;
                    console.log("answer: ", msg);
                    addSentMessage(requestId, timestr, body)
                    addReceivedMessage(requestId, msg);                            
                }                 
            }         
            if(history.length>=1) {
                if(langstate=='korean') {
                    addNotifyMessage("대화를 다시 시작하였습니다.");
                }
                else {
                    addNotifyMessage("Welcome back to the conversation");                               
                }
                chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
            }
        }
    };
    
    var requestObj = {
        "userId": userId,
        "allowTime": allowTime
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

function deleteItems(userId) {
    const uri = "delete";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            console.log("response: " + JSON.stringify(response));

            window.location.href = "index.html";
        }
    };
    
    var requestObj = {
        "userId": userId
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}
