import sys
import pandas as pd
from sklearn.externals import joblib
import s3fs
from io import StringIO
import boto3

def predict(model, data):
    predicted = model.predict(data)
    return predicted


def concatenate_data(identificador, predicted):
    predicted = pd.DataFrame(predicted, columns=['Predicted_prob'])
    data = pd.concat([identificador, predicted], axis=1)
    return data


def evaluate_main(source, model_ser, output):
    data = pd.read_csv(source, encoding='latin')
    model = joblib.load(model_ser)
    identificador = data[['CONTRATO', 'CLIENTE']]
    data = data.drop(['CONTRATO', 'CLIENTE'], axis=1)
    predicted = predict(model, data)
    data = concatenate_data(identificador, predicted)

    if str(output).startswith('s3://'):
        # fs = s3fs.S3FileSystem(key='', secret='')
        s3_bucket = output.split('//')[1].split('/')[0]
        s3_key = output.split(s3_bucket + '/')[1]

        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)

        s3_resource = boto3.resource('s3')
        s3_resource.Object(s3_bucket, s3_key).put(Body=csv_buffer.getvalue())

    else:
        data.to_csv('output/scored_data.csv', index=False)
    return True


if __name__ == '__main__':
    args = len(sys.argv)
    if args < 2:
        raise Exception

    output_file_path = sys.argv[1]

    evaluate_main(source='output/prepared_data.csv',
                  model_ser='models/modelo_fraude_credito_consignado.pkl',
                  output=output_file_path)
