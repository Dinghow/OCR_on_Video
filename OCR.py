# -*- coding: utf-8 -*-
import http.client, urllib.request, urllib.parse, urllib.error, base64, json,os

###############################################
#### Update or verify the following values. ###
###############################################

# Replace the subscription_key string value with your valid subscription key.
subscription_key = ''

uri_base = 'api.cognitive.azure.cn'

headers = {
    # Request headers.
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = urllib.parse.urlencode({
    # Request parameters. The language setting "unk" means automatically detect the language.
    'language': 'zh-Hans',
    'detectOrientation ': 'true',
})

# The source file of a JPEG image containing text.
imgPath = '...\\captions\\' #Please enter the Path by yourself
txtPath = '...\\outTest.txt' #Please enter the Path by yourself
videoName = 'Game of Thrones' #Please enter the video name by yourself

fo = open(txtPath, 'w')
fo.write('The caption of '+videoName+'\n')

#Process the images in a folder
for filename in os.listdir(imgPath):
    f = open(imgPath+filename,'rb')
    body = f.read()

    try:
        # Execute the REST API call and get the response.
        conn = http.client.HTTPSConnection('api.cognitive.azure.cn')
        conn.request("POST", "/vision/v1.0/ocr?%s" % params, body, headers)
        response = conn.getresponse()
        data = response.read()

        # 'data' contains the JSON data. The following formats the JSON data for display.
        parsed = json.loads(data)
        print("Response:")
        print(parsed)
        #Write the data on output file
        fo.write(filename+'\n'+str(parsed))
        conn.close()

    except Exception as e:
        print('Error:')
        print(e)
fo.close()