*** DOWNLOAD PRODUCT ***
1. Download product from: https://github.com/CausalRivers/benchmark/releases/download/First_release/product.zip
2. unzip product
3. rm product.zip

*** INSTALL LIBS ***
python3 -m venv .venv_test
.venv_test\Scripts\activate
OR
source venv_test/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

*** GENERATE DATASET ***
1. .venv_test\Scripts\activate
2. python3 generate_datasets.py
3. deactivate

*** RUN BENCHMARK TEST ***
1. .venv_test\Scripts\activate
2. Update benchmark.yaml
3. python3 benchmark.py
4. deactivate