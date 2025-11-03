cd /var/www/salamamdm
pwd
sudo cd /var/www/salamamdm
sudo cd /var/www
cd /var/www
mkdir /var/www/salamamdm
sudo mkdir /var/www/salamamdm
root
cd..
cd/
az --version
ssh pocadmin@vmpocsalama.southindia.cloudapp.azure.com
ssh pocadmin@135.13.27.176.southindia.cloudapp.azure.com
az vm
az vm list
az --resource-group
list
az --resource-group list
az --resource-group poc_salama
az group list
az account show
az login
mkdir streamlit-app
dir
cd streamlit-app
ls
dir
ls
cd
ls
cat _env
cat _.env
cat .env
cat > startup.sh << 'EOF'
#!/bin/bash
python -m streamlit run app.py --server.port=8000 --server.address=0.0.0.0
EOF

ls
az webapp up --name manufacturing-streamlit-$(date +%s) --runtime "PYTHON:3.11" --sku B1 --location eastus
az provider register --namespace Microsoft.Web
az provider show --namespace Microsoft.Web --query "registrationState"
az webapp up --name manufacturing-streamlit-$(date +%s) --runtime "PYTHON:3.11" --sku B1 --location eastus
ls
git config --global user.name "NaveenKumar"
git config --global user.email "naveenm@systechusa.com"
git init
cat > .gitignore << 'EOF'
_env
*.pyc
__pycache__/
.env
EOF

cat .gitignore
