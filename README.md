# master-thesis

**0. Setup all parameters as you'd like them in setup.py**
You find available lists of tickers in ticker_lists/ or you can create your own.

<br />
<br />

**1. Download stock data from Yahoo Finance**
Downloads stock data into stock_data/

`python download_stock_data.py`
<br />
<br />

**2. Generate stock graphs**
Generates images into stock_graphs/

`python generate_graphs.py`
<br />
<br />

**3. Train the model**
Use optional parameter RESUME if you'd like to resume from a checkpoint. RESUME is a string with the name of the file you want to resume from inside model_training_checkpoints/ . Omit if you want to do normal training.

Finished model is saved into models/ .

`python train_model.py [RESUME]`
<br />
<br />

**4. Test the model**
Tests the model from setup.test_model_name . setup.pretrained_model_name needs to correspond to the same pretrained model as the model that is tested. You find the model name inside models/ .

`python test_model.py`
<br />
<br />
