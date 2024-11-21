# coding=utf-8
# Copyright 2024 Jingze Shi and Bingheng Wu.    All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.



import os
import glob
import shutil
from detect_ma import Yolov5
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import copy
import sys
import hashlib
import hmac
import binascii
from datetime import datetime
import requests
import torch
from pyngrok import ngrok
from pyngrok.conf import PyngrokConfig
import threading

if sys.version_info.major < 3:
    from urllib import quote, unquote

    def hmacsha256(keyByte, message):
        return hmac.new(keyByte, message, digestmod=hashlib.sha256).digest()
    
    # Create a "String to Sign".
    def StringToSign(canonicalRequest, t):
        bytes = HexEncodeSHA256Hash(canonicalRequest)
        return "%s\n%s\n%s" % (Algorithm, datetime.strftime(t, BasicDateFormat), bytes)

else:
    from urllib.parse import quote, unquote

    def hmacsha256(keyByte, message):
        return hmac.new(keyByte.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).digest()

    # Create a "String to Sign".
    def StringToSign(canonicalRequest, t):
        bytes = HexEncodeSHA256Hash(canonicalRequest.encode('utf-8'))
        return "%s\n%s\n%s" % (Algorithm, datetime.strftime(t, BasicDateFormat), bytes)


def urlencode(s):
    return quote(s, safe='~')


def findHeader(r, header):
    for k in r.headers:
        if k.lower() == header.lower():
            return r.headers[k]
    return None


# HexEncodeSHA256Hash returns hexcode of sha256
def HexEncodeSHA256Hash(data):
    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()


# HWS API Gateway Signature
class HttpRequest:
    def __init__(self, method="", url="", headers=None, body=""):
        self.method = method
        spl = url.split("://", 1)
        scheme = 'http'
        if len(spl) > 1:
            scheme = spl[0]
            url = spl[1]
        query = {}
        spl = url.split('?', 1)
        url = spl[0]
        if len(spl) > 1:
            for kv in spl[1].split("&"):
                spl = kv.split("=", 1)
                key = spl[0]
                value = ""
                if len(spl) > 1:
                    value = spl[1]
                if key != '':
                    key = unquote(key)
                    value = unquote(value)
                    if key in query:
                        query[key].append(value)
                    else:
                        query[key] = [value]
        spl = url.split('/', 1)
        host = spl[0]
        if len(spl) > 1:
            url = '/' + spl[1]
        else:
            url = '/'

        self.scheme = scheme
        self.host = host
        self.uri = url
        self.query = query
        if headers is None:
            self.headers = {}
        else:
            self.headers = copy.deepcopy(headers)
        if sys.version_info.major < 3:
            self.body = body
        else:
            self.body = body.encode("utf-8")


BasicDateFormat = "%Y%m%dT%H%M%SZ"
Algorithm = "SDK-HMAC-SHA256"
HeaderXDate = "X-Sdk-Date"
HeaderHost = "host"
HeaderAuthorization = "Authorization"
HeaderContentSha256 = "x-sdk-content-sha256"


# Build a CanonicalRequest from a regular request string
#
# CanonicalRequest =
#  HTTPRequestMethod + '\n' +
#  CanonicalURI + '\n' +
#  CanonicalQueryString + '\n' +
#  CanonicalHeaders + '\n' +
#  SignedHeaders + '\n' +
#  HexEncode(Hash(RequestPayload))
def CanonicalRequest(r, signedHeaders):
    canonicalHeaders = CanonicalHeaders(r, signedHeaders)
    hexencode = findHeader(r, HeaderContentSha256)
    if hexencode is None:
        hexencode = HexEncodeSHA256Hash(r.body)
    return "%s\n%s\n%s\n%s\n%s\n%s" % (r.method.upper(), CanonicalURI(r), CanonicalQueryString(r),
                                       canonicalHeaders, ";".join(signedHeaders), hexencode)


def CanonicalURI(r):
    pattens = unquote(r.uri).split('/')
    uri = []
    for v in pattens:
        uri.append(urlencode(v))
    urlpath = "/".join(uri)
    if urlpath[-1] != '/':
        urlpath = urlpath + "/"  # always end with /
    # r.uri = urlpath
    return urlpath


def CanonicalQueryString(r):
    keys = []
    for key in r.query:
        keys.append(key)
    keys.sort()
    a = []
    for key in keys:
        k = urlencode(key)
        value = r.query[key]
        if type(value) is list:
            value.sort()
            for v in value:
                kv = k + "=" + urlencode(str(v))
                a.append(kv)
        else:
            kv = k + "=" + urlencode(str(value))
            a.append(kv)
    return '&'.join(a)


def CanonicalHeaders(r, signedHeaders):
    a = []
    __headers = {}
    for key in r.headers:
        keyEncoded = key.lower()
        value = r.headers[key]
        valueEncoded = value.strip()
        __headers[keyEncoded] = valueEncoded
        if sys.version_info.major == 3:
            r.headers[key] = valueEncoded.encode("utf-8").decode('iso-8859-1')
    for key in signedHeaders:
        a.append(key + ":" + __headers[key])
    return '\n'.join(a) + "\n"


def SignedHeaders(r):
    a = []
    for key in r.headers:
        a.append(key.lower())
    a.sort()
    return a


# Create the HWS Signature.
def SignStringToSign(stringToSign, signingKey):
    hm = hmacsha256(signingKey, stringToSign)
    return binascii.hexlify(hm).decode()


# Get the finalized value for the "Authorization" header.  The signature
# parameter is the output from SignStringToSign
def AuthHeaderValue(signature, AppKey, signedHeaders):
    return "%s Access=%s, SignedHeaders=%s, Signature=%s" % (
        Algorithm, AppKey, ";".join(signedHeaders), signature)


class Signer:
    def __init__(self):
        self.Key = ""
        self.Secret = ""

    def Verify(self, r, authorization):
        if sys.version_info.major == 3 and isinstance(r.body, str):
            r.body = r.body.encode('utf-8')
        headerTime = findHeader(r, HeaderXDate)
        if headerTime is None:
            return False
        else:
            t = datetime.strptime(headerTime, BasicDateFormat)

        signedHeaders = SignedHeaders(r)
        canonicalRequest = CanonicalRequest(r, signedHeaders)
        stringToSign = StringToSign(canonicalRequest, t)
        return authorization == SignStringToSign(stringToSign, self.Secret)

    # SignRequest set Authorization header
    def Sign(self, r):
        if sys.version_info.major == 3 and isinstance(r.body, str):
            r.body = r.body.encode('utf-8')
        headerTime = findHeader(r, HeaderXDate)
        if headerTime is None:
            t = datetime.utcnow()
            r.headers[HeaderXDate] = datetime.strftime(t, BasicDateFormat)
        else:
            t = datetime.strptime(headerTime, BasicDateFormat)

        haveHost = False
        for key in r.headers:
            if key.lower() == 'host':
                haveHost = True
                break
        if not haveHost:
            r.headers["host"] = r.host
        signedHeaders = SignedHeaders(r)
        canonicalRequest = CanonicalRequest(r, signedHeaders)
        stringToSign = StringToSign(canonicalRequest, t)
        signature = SignStringToSign(stringToSign, self.Secret)
        authValue = AuthHeaderValue(signature, self.Key, signedHeaders)
        r.headers[HeaderAuthorization] = authValue
        r.headers["content-length"] = str(len(r.body))
        queryString = CanonicalQueryString(r)
        if queryString != "":
            r.uri = r.uri + "?" + queryString


class yolov5_detection():
    def __init__(self, model_name, model_path):
        print('model_name:',model_name)
        print('model_path:',model_path)
                
        self.model = Yolov5(model_path, device='cuda:0')
        
        self.capture = "test.png"

    def _preprocess(self, data):
       
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)
        return "ok"
    
    def _inference(self, data):
        pred_result = self.model.inference(self.capture)
        return pred_result

    def _postprocess(self, data):
        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []

        for pred in data:
            classes, _, x1, y1, x2, y2, conf = pred
            detection_classes.append(classes)
            boxes = [y1,x1,y2,x2]
            detection_boxes.append(boxes)
            detection_scores.append(conf)
                
        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores
     
        return result


app = fastapi.FastAPI(
    debug=True,
    title="Cheems",
    version="0.1.0"
)


origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.abspath(os.path.dirname(__file__))

@app.get("/")
def read_root():
    # 返回200状态码
    return {"code": 200, "message": "Hello, Cheems!"}


class UploadImage(BaseModel):
    image: str

# 用于接收前端传来的图片
@app.post("/upload")
def upload_image(data: UploadImage):
    image = base64.b64decode(data.image)
    with open(ROOT + "/test.png", "wb") as f:
        f.write(image)
    return "upload success"


class InferRequest(BaseModel):
    # 接受方式
    infer_type: str


SERVICE = yolov5_detection("yolo", "model\yolo.pt")
@app.post("/infer")
def infer(data: InferRequest):
    if data.infer_type == "huawei":
        # 华为云的认证
        method = "POST"
        url = "https://infer-modelarts-cn-southwest-2.myhuaweicloud.com/v1/infers/1630faa3-aa12-435d-88ee-565f5325639e"
        headers = {"x-sdk-content-sha256": "UNSIGNED-PAYLOAD"}
        request = HttpRequest(method, url, headers)
        sig = Signer()
        sig.Key = "2CJFVCRXVGSPM4HCNGMV"
        sig.Secret = "enF7EcrQ2c5J0euaM8WCjfd135pHCJxBkR2SQM2E"
        sig.Sign(request)

        
        # 推理
        file = {"images": open(ROOT + "/test.png", "rb")} 
        response = requests.request(request.method, request.scheme + "://" + request.host + request.uri, headers=request.headers, files=file).json()

        # 标注
        image = Image.open(ROOT + "/test.png")
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(64)
        for i in range(len(response['detection_classes'])):
            # y1,x1,y2,x2 -> x1,y1,x2,y2
            box = response['detection_boxes'][i]
            box = [box[1], box[0], box[3], box[2]]
            draw.rectangle(box, outline="black", width=4)
            draw.text((box[0], box[1]), f"{response['detection_classes'][i]}:{response['detection_scores'][i]:.2f}", fill="black", font=font)
        image.save(ROOT + "/test.png")
        # 生成一个id并保存图片与json标注
        id = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        image.save(ROOT + f"/output/{id}.png")
        with open(ROOT + f"/output/{id}.json", "w") as f:
            json.dump(response, f)
        with open(ROOT + "/test.png", "rb") as f:
            response['image_label'] = base64.b64encode(f.read()).decode()
        
        return Response(content=json.dumps(response), media_type="application/json")
    else:
        
        # 推理
        pred_result = SERVICE._inference(123)
        # 后处理
        result = SERVICE._postprocess(pred_result)
        # 标注
        image = Image.open(ROOT + "/test.png")
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(64)
        for i in range(len(result['detection_classes'])):
            # y1,x1,y2,x2 -> x1,y1,x2,y2
            box = result['detection_boxes'][i]
            box = [box[1], box[0], box[3], box[2]]
            draw.rectangle(box, outline="black", width=4)
            draw.text((box[0], box[1]), f"{result['detection_classes'][i]}:{result['detection_scores'][i]:.2f}", fill="black", font=font)
        image.save(ROOT + "/test.png")
        # 生成一个id并保存图片与json标注
        id = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        image.save(ROOT + f"/output/{id}.png")
        with open(ROOT + f"/output/{id}.json", "w") as f:
            json.dump(result, f)
        with open(ROOT + "/test.png", "rb") as f:
            result['image_label'] = base64.b64encode(f.read()).decode()
        # 清除cuda缓存
        torch.cuda.empty_cache()

        return Response(content=json.dumps(result), media_type="application/json")

def public_url():
    ngrok.set_auth_token("2NOOvNoVKONBwuqYBSCZYFgj7v8_3JBQUkru7Fre6NXvbcowL")
    config = PyngrokConfig(region='ap')
    public_url = ngrok.connect(8080, pyngrok_config=config)
    print(public_url)


if __name__ == "__main__":

    threading.Thread(target=public_url).start()


    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
    
