import requests
import os

def download_latest_rulekit():
    print("Fetching latest RuleKit release from GitHub...")
    response = requests.get('https://api.github.com/repos/adaa-polsl/RuleKit/releases/latest')
    response.raise_for_status()
    release_data = response.json()
    
    jar_url = None
    for asset in release_data.get('assets', []):
        if asset['name'].endswith('.jar'):
            jar_url = asset['browser_download_url']
            break
            
    if not jar_url:
        print("Could not find a .jar asset in the latest release.")
        return
        
    os.makedirs('bin', exist_ok=True)
    jar_path = os.path.join('bin', 'rulekit.jar')
    
    print(f"Downloading from {jar_url}...")
    jar_response = requests.get(jar_url)
    jar_response.raise_for_status()
    
    with open(jar_path, 'wb') as f:
        f.write(jar_response.content)
        
    print(f"Successfully downloaded RuleKit to {jar_path}")

if __name__ == "__main__":
    download_latest_rulekit()
