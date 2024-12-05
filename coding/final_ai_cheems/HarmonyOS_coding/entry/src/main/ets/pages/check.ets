<!-- coding=utf-8
Copyright 2024 Jingze Shi and Bingheng Wu.    All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License. -->



import http from '@ohos.net.http';
import router from '@ohos.router';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { fileIo as fs, fileUri } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';
import { buffer } from '@kit.ArkTS';

@Entry
@Component
struct Index {
  private context = getContext(this);
  @State message: string = 'Thank you Cheems!';
  @State message_font_size: number = 24;
  @State image_path: string = 'file://com.shijingze.last_cheems/data/storage/el2/base/haps/entry/files/test.jpg';
  @State URL: string = 'https://3197-124-93-196-27.ngrok-free.app';
  @State TYPE: string = 'local'; //  huawei or local



  build() {
    RelativeContainer() {

      Column() {
        Image(this.image_path)
          .id('Image')
          .width(480)
          .height(320)
          .alignRules({
            top: { anchor: '__container__', align: VerticalAlign.Top },
            middle: { anchor: '__container__', align: HorizontalAlign.Center }
          })

        Scroll() {
          Column({ space:10000}) {
            Text(this.message)
              .fontSize(this.message_font_size)
              .fontWeight(FontWeight.Bold)
              .alignRules({
                top: { anchor: '__container__', align: VerticalAlign.Top },
                middle: { anchor: '__container__', align: HorizontalAlign.Center }
              })
          }
        }
        .width("100%")
        .height(320)
        .scrollable(ScrollDirection.Vertical)
        .padding({
          bottom: 300
        })

        Row() {

        Button('上传')
          .id('update_image')
          .fontSize(24)//  深灰背景
          .backgroundColor('#333333')
          .fontWeight(FontWeight.Bold)
          .margin({
            bottom: 10
          })//  在Text下方
          .alignRules({
            bottom: { anchor: '__container__', align: VerticalAlign.Bottom },
            middle: { anchor: '__container__', align: HorizontalAlign.Center }
          })
          .onClick(() => {
            const photoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
            photoSelectOptions.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE; // 过滤选择媒体文件类型为IMAGE
            photoSelectOptions.maxSelectNumber = 1; // 选择媒体文件的最大数目
            let uris: Array<string> = [];
            const photoViewPicker = new photoAccessHelper.PhotoViewPicker();
            photoViewPicker.select(photoSelectOptions)
              .then((photoSelectResult: photoAccessHelper.PhotoSelectResult) => {
                uris = photoSelectResult.photoUris;
                let file = fs.openSync(uris[0]);
                let file_arrayBuffer = new ArrayBuffer(6291456);
                let len = fs.readSync(file.fd, file_arrayBuffer);
                let base64 = new util.Base64Helper();
                // 将arrayBuffer转成Uint8Array
                let file_data = new Uint8Array(file_arrayBuffer);
                let httpRequest = http.createHttp();
                httpRequest.request(this.URL + '/upload', {
                  method: http.RequestMethod.POST,
                  header: {
                    'Content-Type': 'application/json'
                  },
                  extraData: {
                    "image": base64.encodeToStringSync(file_data)
                  }
                }, async (err: Error, data: http.HttpResponse) => {
                  if (err) {
                    console.log('error:' + JSON.stringify(err));
                  } else {
                    console.log('response:' + JSON.stringify(data));
                    let random_name = Math.random().toString(36).substring(2);
                    let save_uri = fileUri.getUriFromPath(this.context.filesDir + '/' + random_name + '.jpg');
                    let save_image = fs.openSync(save_uri, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
                    let len = fs.writeSync(save_image.fd, file_arrayBuffer);
                    console.log('len:' + len)
                    this.image_path = save_uri
                    console.log('save_uri:' + save_uri)
                    this.message = "图片上传成功"
                    // 关闭文件
                    fs.closeSync(save_image.fd);
                    fs.closeSync(file.fd);
                  }
                })
              });
          });

          Button('检查')
            .id('into_quality_check')
            .fontSize(24)//  深灰背景
            .backgroundColor('#333333')
            .fontWeight(FontWeight.Bold)
            .margin({
              bottom: 10
            })//  在Text下方
            .alignRules({
              bottom: { anchor: '__container__', align: VerticalAlign.Bottom },
              middle: { anchor: '__container__', align: HorizontalAlign.Center }
            })
            .onClick(() => {
              let httpRequest = http.createHttp();
              httpRequest.request(this.URL + '/infer', {
                method: http.RequestMethod.POST,
                header: {
                  'Content-Type': 'application/json'
                },
                extraData: {
                  "infer_type": this.TYPE
                },
                maxLimit: 104857600
              }, async (err, data) => {
                if (err) {
                  console.log('error:' + JSON.stringify(err));
                } else {
                  console.log('response:' + JSON.stringify(data));
                  let base64 = new util.Base64Helper();

                  interface DetectionResult {
                    detection_classes: string[];
                    detection_boxes: number[][];
                    detection_scores: number[];
                    image_label: string;
                  }
                  let parse_data: DetectionResult = JSON.parse(data.result.toString());
                  let image_label = base64.decodeSync(parse_data.image_label)
                  let image_Buffer = buffer.from(image_label)

                  let rand_name = Math.random().toString(36).substring(2);
                  let save_uri = fileUri.getUriFromPath(this.context.filesDir + '/' + rand_name + '.jpg');
                  let save_image = fs.openSync(save_uri, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
                  let len = fs.writeSync(save_image.fd, image_Buffer.buffer);
                  console.log('len:' + len)
                  this.image_path = save_uri
                  console.log('save_uri:' + save_uri)
                  this.message = JSON.stringify(parse_data.detection_classes) + "\n\n" + JSON.stringify(parse_data.detection_boxes) + "\n\n" + JSON.stringify(parse_data.detection_scores)
                  this.message_font_size = 16
                }
              })
            })
        }

        Button('返回初始页面')
          .id('intoQualityCheck')
          .fontSize(24)//  深灰背景
          .height(40)
          .backgroundColor('#333333')
          .fontWeight(FontWeight.Bold)//  在Text下方
          .margin({
            top: 10
          })
          .alignRules({
            bottom: { anchor: '__container__', align: VerticalAlign.Bottom },
            middle: { anchor: '__container__', align: HorizontalAlign.Center }
          })// 跳转到质检页面page/check.ets
          .onClick(() => {
            router.pushUrl({
              url: 'pages/Index'
            })
          })



      }

    }
  }
}