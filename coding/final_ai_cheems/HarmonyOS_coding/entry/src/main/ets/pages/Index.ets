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



import router from '@ohos.router';

@Entry
@Component
struct Index {
  @State message: string = 'Thank you Cheems!';

  build() {
    Row() {
      Column() {
        Image($r("app.media.huawei_logo_1"))
          .width(100)
        Text('  Cheems的钛合金激光眼  ')
          .fontColor(Color.White)
          .backgroundColor(Color.Brown)
          .height(100)
          .fontSize(32)
          .fontWeight(700)
        Text('\n' +
          '《Cheems的钛合金激光眼》是一款专为PCB板工业品检测设计的鸿蒙应用程序')
          .lineHeight(24)
          .textOverflow({ overflow: TextOverflow.MARQUEE })
          .fontColor(Color.Black)
            //.backgroundColor('#fff5f1f1')
          .fontSize(20)
          .fontWeight(400)

        Row() {
          Text('团队:Thank you Cheems!')
            .fontColor(Color.Black)
              //.backgroundColor('#a1a1a1')
            .fontSize(20)
            .height(200)
            // 宽度为100%
            .width(240)
            .fontWeight(700)
            .alignRules({
              middle: { anchor: '__container__', align: HorizontalAlign.Center }
            })
        }
        .alignRules({
          top: { anchor: '__container__', align: VerticalAlign.Top },
          middle: { anchor: '__container__', align: HorizontalAlign.Center }
        })

        Button('进入质检页面')
          .id('intoQualityCheck')
          .fontSize(24)
          .height(40)
            //  深灰背景
          .backgroundColor('#333333')
          .fontWeight(FontWeight.Bold)
          .margin({
            top: 240,
            bottom: 0
          })
          .alignRules({
            bottom: { anchor: '__container__', align: VerticalAlign.Bottom },
            middle: { anchor: '__container__', align: HorizontalAlign.Center }
          })
          .onClick(() => {
            router.pushUrl({
              url: 'pages/check'
            })
          })
      }

      .width('100%')
    }
    .height('100%')
  }
}
