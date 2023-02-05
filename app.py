from flask import Flask, jsonify, render_template, request
from flask_swagger import swagger
import re
from quality import image_quality_detector as imqd
from sensitivity import sensitive_data_cnn as sd
from sensitivity import sensitive_data_pre as dp
from sensitivity import sensitive_image_data as sid
from quality import quality_data as dq



app = Flask(__name__)

@app.route("/")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "v.1.0"
    swag['info']['title'] = "Welcome to Datatera Beta"
    return jsonify(swag)


@app.route('/api')
def get_api():
    return render_template('swaggerui.html')


# Check Credit Card Number
@app.route("/beta/checkcreditcardno/<string:candidate_value>", methods=["GET"], endpoint='check_credit_card_no')
def check_credit_card_no(candidate_value):
    candidate_value = re.sub('\D', '', str(candidate_value))
    x = re.search(
        "^(?:4[0-9]{12}(?:[0-9]{3})?|[25][1-7][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$",
        candidate_value)

    if x:
        return jsonify(f"Credit Card Number is detected!!")
    else:
        return jsonify(f"No Credit Card Number is detected!!")


@app.route("/beta/sensitivedatacnn", methods=["GET"], endpoint='sensitive_data_cnn')
def sensitive_data_cnn():
    response = sd.sensitive_data_cnn()

    return jsonify(f"Result Accuracy: {response.url_accuracy} Result Loss: {response.url_loss}")


@app.route("/beta/sensitivedatapre", methods=["GET"], endpoint='sensitive_data_pre')
def sensitive_data_pre():
    url = request.args.get('Anonymize_Data')
    data = dp.sensitive_data_pre(url)
    return jsonify(f"Anonymized/Nonanonymized Data:{data}")


@app.route("/beta/qualitydata", methods=["GET"], endpoint='quality_data')
def quality_data():
    return dq.quality_data()


# get images quality matrixes
@app.route("/beta/imagequalitydetector", methods=["GET"], endpoint='imagequalitydetector')
def imagequalitydetector():
    images = imqd.getQualityMatrixes()
    return jsonify(images)

@app.route("/beta/imagequalitysensivity", methods=["GET"], endpoint='imagequalitysensivity')
def imagequalitysensivity():
    path = request.args.get('path')
    #print('pathnew')
    #print(path)
    result = sid.getImageSensitiveData(path)
    print('result')
    print(result)
    return result
