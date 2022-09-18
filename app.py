from shipment.entity.predictor import ShipmentPredictor, ShipmentData
from shipment.constant import CONFIG_DIR, get_current_time_stamp
from shipment.utils.util import read_yaml_file, write_yaml_file
from shipment.config.configuration import Configuration
from flask import send_file, abort, render_template
from shipment.pipeline.pipeline import Pipeline
from shipment.logger import get_log_dataframe
from shipment.logger import logging
from flask import Flask, request
import os
import json

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "shipment"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

CONCRETE_DATA_KEY = "shipment_data"
CONCRETE_STRENGTH_VALUE_KEY = "shipment_strength_value"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'shipment'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("shipment", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path)
             if "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        CONCRETE_DATA_KEY: None,
        CONCRETE_STRENGTH_VALUE_KEY: None
    }

    if request.method == 'POST':
        country = str(request.form['country'])
        Managed_By = str(request.form['Managed_By'])
        Fulfill_Via = str(request.form['Fulfill_Via'])
        Vendor_INCO_Term = str(request.form['Vendor_INCO_Term'])
        Shipment_Mode = str(request.form['Shipment_Mode'])
        Scheduled_Delivery_Date = str(request.form['Scheduled_Delivery_Date'])
        Delivered_to_Client_Date = str(request.form['Delivered_to_Client_Date'])
        Delivery_Recorded_Date = str(request.form['Delivery_Recorded_Date'])
        Product_Group = str(request.form['Product_Group'])
        Sub_Classification = str(request.form['Sub_Classification'])
        Vendor = str(request.form['Vendor'])
        Item_Description = str(request.form['Item_Description'])
        Molecule_Test_Type = str(request.form['Molecule_Test_Type'])
        Brand = str(request.form['Brand'])
        Dosage = str(request.form['Dosage'])
        Dosage_Form = str(request.form['Dosage_Form'])
        Unit_of_Measure_Per_Pack = int(request.form['Unit_of_Measure_Per_Pack'])
        Line_Item_Quantity = int(request.form['Line_Item_Quantity'])
        Line_Item_Value = float(request.form['Line_Item_Value'])
        Pack_Price = float(request.form['Pack_Price'])
        Manufacturing_Site = str(request.form['Manufacturing_Site'])
        First_Line_Designation = str(request.form['First_Line_Designation'])
        Line_Item_Insurance_USD = float(request.form['Line_Item_Insurance_USD'])

        shipment_data = ShipmentData(
            country=country,
            Managed_By=Managed_By,
            Fulfill_Via=Fulfill_Via,
            Vendor_INCO_Term=Vendor_INCO_Term,
            Shipment_Mode=Shipment_Mode,
            Scheduled_Delivery_Date=Scheduled_Delivery_Date,
            Delivered_to_Client_Date=Delivered_to_Client_Date,
            Delivery_Recorded_Date=Delivery_Recorded_Date,
            Product_Group=Product_Group,
            Sub_Classification=Sub_Classification,
            Vendor=Vendor,
            Item_Description=Item_Description,
            Molecule_Test_Type=Molecule_Test_Type,
            Brand=Brand,
            Dosage=Dosage,
            Dosage_Form=Dosage_Form,
            Unit_of_Measure_Per_Pack=Unit_of_Measure_Per_Pack,
            Line_Item_Quantity=Line_Item_Quantity,
            Line_Item_Value=Line_Item_Value,
            Pack_Price=Pack_Price,
            Manufacturing_Site=Manufacturing_Site,
            First_Line_Designation=First_Line_Designation,
            Line_Item_Insurance_USD=Line_Item_Insurance_USD
        )
        shipment_df = shipment_data.get_shipment_input_data_frame()
        shipment_predictor = ShipmentPredictor(model_dir=MODEL_DIR)
        median_shipment_value = shipment_predictor.predict(X=shipment_df)
        context = {
            CONCRETE_DATA_KEY: shipment_data.get_shipment_input_data_frame(),
            CONCRETE_STRENGTH_VALUE_KEY: median_shipment_value,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()
