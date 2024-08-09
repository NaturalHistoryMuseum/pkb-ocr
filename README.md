### 
PKB OCR Service:
### 
http://3.8.176.21:8501/

### 
sudo apt update
### 
sudo apt-get update
### 
sudo apt upgrade -y
### 
sudo apt install git curl unzip tar make sudo vim wget -y
### 
git clone "Your-repository"
### 
sudo apt install python3-pip
### 
python3 -m venv myenv
### 
source myenv/bin/activate
### 
pip install -r requirements.txt
### 
pip install 'awswrangler[gremlin, opencypher, sparql]'

### 
#Temporary running
### 
python3 -m streamlit run st_pkb.py
### 
#Permanent running
### 
nohup python3 -m streamlit run st_pkb.py
