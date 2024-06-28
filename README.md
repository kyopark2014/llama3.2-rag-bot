# Llama3로 RAG를 구현하는 Workshop

Llama3를 이용해 RAG를 구현합니다. 전체적인 Architecture는 아래와 같습니다.

<img src="./images/basic-architecture.png" width="800">


## Architecture의 구현 

### LangChain

```python
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }
    )
)
parameters = {
    "max_gen_len": 1024,  
    "top_p": 0.9, 
    "temperature": 0.1
}    

chat = ChatBedrock(   
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
)
```

### Basic Chat

```python
def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다. 답변은 한국어로 합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    
    history = memory_chain.load_memory_variables({})["chat_history"]
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        
    return msg
```

여기서 Stream은 아래와 같이 event를 추출하여 json format으로 client에 결과를 전달합니다. 

```python
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            sendMessage(connectionId, result)
    return msg
```

### 대화 이력의 관리

사용자가 접속하면, DynamoDB에서 대화 이력을 가져옵니다. 이것은 최초 접속 1회만 수행합니다. 

```python
def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )


```

Context에 넣을 history를 가져와서 memory_chain에 등록합니다.

```pytho
for item in response['Items']:
    text = item['body']['S']
    msg = item['msg']['S']
    type = item['type']['S']

    if type == 'text' and text and msg:
        memory_chain.chat_memory.add_user_message(text)
        if len(msg) > MSG_LENGTH:
            memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
        else:
            memory_chain.chat_memory.add_ai_message(msg) 
```

Lambda와 같은 서버리스는 이벤트가 있을 경우에만 사용이 가능하므로, 이벤트의 userId를 기준으로 메모리를 관리합니다. 

map_chain = dict()

```python
if userId in map_chain:  
    memory_chain = map_chain[userId]    
else: 
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer’,
              return_messages=True, k=10)
    map_chain[userId] = memory_chain
```

새로운 입력(text)과 응답(msg)를 user/ai message로 저장합니다.

```python
memory_chain.chat_memory.add_user_message(text)
memory_chain.chat_memory.add_ai_message(msg)
```

### WebSocket Stream 사용하기 

#### Client 동작

WebSocket을 연결하기 위하여 endpoint를 접속을 수행합니다. onmessage()로 메시지를 받습니다. WebSocket이 연결되면 onopen()로 초기화를 수행합니다. 일정 간격으로 keep alive 동작을 수행합니다. 네트워크 재접속 등의 이유로 세션이 끊어지면 onclose()로 확인할 수 있습니다.

```python
const ws = new WebSocket(endpoint);
ws.onmessage = function (event) {        
    response = JSON.parse(event.data)

    if(response.request_id) {
        addReceivedMessage(response.request_id, response.msg);
    }
};
ws.onopen = function () {
    isConnected = true;
    if(type == 'initial')
        setInterval(ping, 57000); 
};
ws.onclose = function () {
    isConnected = false;
    ws.close();
};
```

발신 메시지는 JSON 포맷으로 아래와 같이 userId, 요청시간, 메시지 타입과 메시지를 포함합니다. 발신시 WebSocket의 send()을 이용하여 발신합니다. 발신시점에 세션이 연결되어 있지 않다면 연결하고 재시도하도록 알림을 표시합니다.

```python
sendMessage({
    "user_id": userId,
    "request_id": requestId,
    "request_time": requestTime,        
    "type": "text",
    "body": message.value
})
WebSocket = connect(endpoint, 'initial');
function sendMessage(message) {
    if(!isConnected) {
        WebSocket = connect(endpoint, 'reconnect');        
        addNotifyMessage("재연결중입니다. 잠시후 다시시도하세요.");
    }
    else {
        WebSocket.send(JSON.stringify(message));     
    }     
}
```

#### Server 동작

Client로 부터 메시지 수신은 Lambda로 전달된 event에서 connectionId와 routeKey를 이용해 수행합니다. 이때 keep alive 동작을 수행하여 세션을 유지합니다. 메시지 발신은 boto3로 "apigatewaymanagementapi"로 client를 정의한 후에 client.post_to_connection()로 전송합니다.

```python
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi’,      
        endpoint_url=connection_url)

def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

def lambda_handler(event, context):
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            if body[0:8] == "__ping__":  # keep alive
                sendMessage(connectionId, "__pong__")
            else:
                msg, reference = getResponse(connectionId, jsonBody)
```

### Prompt 사용 예: 번역하기

Prompt Engineering을 이용하여 손쉽게 한/영 번역을 수행합니다.

```python
def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to 
         {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
```

Input/output 언어 타입과 입력 텍스트를 지정 후 chain.invoke()를 이용합니다.

```python
    chain = prompt | chat    
    result = chain.invoke(
        {
            "input_language": input_language,
            "output_language": output_language,
            "text": text,
        }
    )
        
    msg = result.content
    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
```


### Prompt 사용 예: 문법 오류 고치기

Prompt Engineering을 이용해서 한/영 문법 오류 고치는 API를 만들 수 있습니다.

```python
def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```    

### Prompt 사용 예: 코드 요약하기

Prompt Engineering을 이용해서 코드 요약하기 API를 만들 수 있습니다.

```python
def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```

### RAG

RAG에서는 <context> tag를 이용해 Relevant Documents를 넣도록  Prompt를 구성합니다. 

```python
def query_using_RAG_context(connectionId, requestId, chat, context, revised_question):    
    system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```

History를 이용한 revised question과 Stream을 활용해서 성능 및 사용성을 높입니다.

```python
    chain = prompt | chat
    
    stream = chain.invoke(
        {
            "context": context,
            "input": revised_question,
        }
    )
    msg = readStreamMsg(connectionId, requestId, 
            stream.content)    

    return msg
```

OpenSearch를 이용해 Vector Store를 정의하고, 읽어온 문서를 등록합니다.

```python
def store_document_for_opensearch(bedrock_embeddings, docs, documentId):
        delete_document_if_exist(metadata_key)

        vectorstore = OpenSearchVectorSearch(
            index_name=index_name,  
            is_aoss = False,
            #engine="faiss",  # default: nmslib
            embedding_function = bedrock_embeddings,
            opensearch_url = opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd),
        )
        response = vectorstore.add_documents(docs, bulk_size = 2000)
```

Vectorstore를 통해 관련된 문서를 추출하여 context로 활용합니다.

```python
# vector search (semantic) 
    relevant_documents = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k,
    )
relevant_docs = [] 
if(len(rel_docs)>=1):
        for doc in rel_docs:
            relevant_docs.append(doc)

    for document in relevant_docs:
        content = document['metadata']['excerpt']
                
        relevant_context = relevant_context + content + "\n\n"

msg = query_using_RAG_context(connectionId, requestId, chat, relevant_context, revised_question)
```

### RAG의 Parent/Child Chunking

문서를 크기에 따라 parent chunk와 child chunk로 나누어서 child chunk를 찾은 후에 LLM의 context에는 parent chunk를 사용하면, 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다.

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    # separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
```

Parent/Child Chunking을 수행하는 과정은 아래와 같습니다. 

1) parent/child로 chunking을 수행합니다.

2) parent doc을 OpenSearch에 add하면, parent_doc_id가 생성됩니다. 

3) child doc의 meta에 parent_doc_id를 등록합니다.
  
4) 문서 검색시, 필터를 이용해 child 문서를 검색합니다.
  
5) 검색된 child 문서들이 parent가 동일하다면 중복을 제거합니다.
  
6) parent_doc_id를 이용하여 OpenSearch에서 parent doc을 가져와 context로 활용합니다. 

Parent chunk의 meta에 “doc_level”을 “parent”로 지정하고 OpenSearch에 등록합니다. 

```python
parent_docs = parent_splitter.split_documents(docs)
    if len(parent_docs):
        for i, doc in enumerate(parent_docs):
            doc.metadata["doc_level"] = "parent"
                    
        parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
```

Child chunk의 meta에 “doc_level”을 “child”로 지정하고 “parent_doc_id”로 parent chunk의 document id를 지정합니다. 

```python                
        child_docs = []
        for i, doc in enumerate(parent_docs):
            _id = parent_doc_ids[i]
            sub_docs = child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata["parent_doc_id"] = _id
                _doc.metadata["doc_level"] = "child"
            child_docs.extend(sub_docs)
                
        child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)
                    
        ids = parent_doc_ids+child_doc_ids
```

OpenSearch에 RAG 정보를 요청할 때에 아래와 같이 pre_filter로 doc_level이 child인 문서들을 검색합니다. 

```python
def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        pre_filter={"doc_level": {"$eq": "child"}}
    )
            
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
```

Child chunk의 parent_doc_id가 중복이 아닌 경우만 relevant_document로 활용합니다. 

```python
      
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                    
                    if len(relevant_documents)>=top_k:
                        break
                                
return relevant_documents
```

OpenSearch에서 parent doc의 가져와서 RAG에서 활용합니다.

```python
relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, keyword, top_k)

for i, document in enumerate(relevant_documents):
    parent_doc_id = document[0].metadata['parent_doc_id']
    doc_level = document[0].metadata['doc_level']        
    excerpt, uri = get_parent_document(parent_doc_id) # use pareant document

def get_parent_document(parent_doc_id):
    response = os_client.get(
        index="idx-rag", 
        id = parent_doc_id
    )
    
    source = response['_source']                                
    metadata = source['metadata']    
    return source['text'], metadata['uri']
```

Meta 파일을 생성하면 문서 업데이트나 삭제시 유용하게 사용할 수 있습니다.

```python
def create_metadata(bucket, key, meta_prefix, s3_prefix, uri, category, documentId, ids):
    title = key
    timestamp = int(time.time())

    metadata = {
        "Attributes": {
            "_category": category,
            "_source_uri": uri,
            "_version": str(timestamp),
            "_language_code": "ko"
        },
        "Title": title,
        "DocumentId": documentId,      
        "ids": ids  
    }
    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])

    client = boto3.client('s3')
    try: 
        client.put_object(
            Body=json.dumps(metadata), 
            Bucket=bucket, 
            Key=meta_prefix+objectName+'.metadata.json' 
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")
```

문서를 삭제하거나 업데이트 할 때에 OpenSearch의 문서를 삭제합니다. 

```python
def delete_document_if_exist(metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            
            ids = json.loads(meta)['ids']

            result = vectorstore.delete(ids) 
        else:
            print('no meta file: ', metadata_key)
```

### RAG의 파일 업로드

S3에 Object 업로드시 발생하는 이벤트 형태에는 OBJECT_CREATED_PUT (일반파일), CREATED_COMPLETE_MULTIPART_UPLOAD (대용량 파일)이 있습니다.

```python
const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
    events: [
      s3.EventType.OBJECT_CREATED_PUT,
      s3.EventType.OBJECT_REMOVED_DELETE,
      s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
    ],
    filters: [
      { prefix: s3_prefix+'/' },
    ]
  });
  lambdaS3eventManager.addEventSource(s3PutEventSource);
```

### RAG의 결과를 신뢰도에 따라 정렬하기

FAISS를 이용해 일정 신뢰도 이상만을 관련된 문서로 활용합니다. 

```python
if len(relevant_docs) >= 1:
    selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embeddings)

def priority_search(query, relevant_docs, bedrock_embeddings):
    excerpts = []
    for i, doc in enumerate(relevant_docs):
        excerpts.append(
            Document(
                page_content=doc['metadata']['excerpt'],
                metadata={
                    'name': doc['metadata']['title'],
                    'order':i,
                }
            )
        )  

    embeddings = bedrock_embeddings
    vectorstore_confidence = FAISS.from_documents(
        excerpts,  # documents
        embeddings  # embeddings
    )            
    rel_documents = 
        vectorstore_confidence.similarity_search_with_score(
             query=query,
             k=top_k
        )
    docs = []
    for i, document in enumerate(rel_documents):
        order = document[0].metadata['order']
        name = document[0].metadata['name']
        assessed_score = document[1]

        relevant_docs[order]['assessed_score'] = int(assessed_score)

        if assessed_score < 200:
            docs.append(relevant_docs[order])    
    return docs
```



