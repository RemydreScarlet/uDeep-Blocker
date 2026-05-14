import os
import re
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class UltimateJSCollector:
    def __init__(self, root="dataset"):
        self.root = root
        self.hashes = set()
        os.makedirs(f"{root}/ad_tracker", exist_ok=True)
        os.makedirs(f"{root}/clean", exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (TrackerDetector-Research)"})

    def get_hash(self, content):
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def extract_urls(self, text):
        """EasyList/Adblock形式からJSのURLを抽出"""
        found = re.findall(r"\|\|([a-z0-9.-]+\/.*\.js)", text)
        return [f"https://{p.split('$')[0]}" for p in found]

    def get_all_cdnjs_urls(self, limit=1000):
        """cdnjsから全ライブラリの全バージョンのURLを取得"""
        print(f"Fetching library list from cdnjs (Limit: {limit})...")
        try:
            libs = self.session.get("https://api.cdnjs.com/libraries").json()['results']
        except Exception as e:
            print(f"Failed to fetch cdnjs list: {e}")
            return []

        all_files = []
        # 進捗バー付きでメタデータを取得
        for lib in tqdm(libs[:limit], desc="Scanning cdnjs metadata"):
            try:
                detail = self.session.get(f"https://api.cdnjs.com/libraries/{lib['name']}", timeout=10).json()
                for asset in detail.get('assets', []):
                    version = asset['version']
                    for file in asset['files']:
                        if file.endswith('.js'):
                            url = f"https://cdnjs.cloudflare.com/ajax/libs/{lib['name']}/{version}/{file}"
                            all_files.append(url)
            except:
                continue
        return all_files

    def download_and_save(self, url, label):
        try:
            # URLからパラメータを排除
            clean_url = url.split('$')[0]
            res = self.session.get(clean_url, timeout=5)
            
            # HTML（404ページなど）ではなく、ある程度のサイズがあるJSのみ保存
            if res.status_code == 200 and len(res.text) > 200 and not(res.text.strip().startswith('<')):
                h = self.get_hash(res.text)
                if h not in self.hashes:
                    self.hashes.add(h)
                    with open(os.path.join(self.root, label, f"{h}.js"), "w", encoding="utf-8") as f:
                        f.write(res.text)
        except:
            pass

    def run(self):
        # 1. 広告・トラッカーの収集
        sources = {
            "tracking": "https://malware-filter.gitlab.io/malware-filter/tracking-filter.txt",
            "easylist": "https://easylist.to/easylist/easylist.txt",
            "uBlock_Unbreak": "https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/unbreak.txt",
            "AdGuard_Tracking": "https://raw.githubusercontent.com/AdguardTeam/AdguardFilters/master/SpywareFilter/sections/tracking_servers.txt",
            "EasyPrivacy": "https://easylist.to/easylist/easyprivacy.txt",
        }
        
        ad_urls = []
        for name, url in sources.items():
            print(f"Fetching {name}...")
            try:
                raw = self.session.get(url).text
                ad_urls.extend(self.extract_urls(raw))
            except:
                print(f"Failed to fetch {name}")

        ad_urls = list(set(ad_urls))
        print(f"Total unique Ad/Tracker URLs: {len(ad_urls)}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(tqdm(executor.map(lambda u: self.download_and_save(u, "ad_tracker"), ad_urls), 
                      total=len(ad_urls), desc="Downloading Ad/Tracker JS"))

        # 2. クリーンなデータの収集 (cdnjs Full Archive)
        clean_urls = self.get_all_cdnjs_urls(limit=100)
        print(f"Total unique Clean JS URLs: {len(clean_urls)}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(tqdm(executor.map(lambda u: self.download_and_save(u, "clean"), clean_urls), 
                      total=len(clean_urls), desc="Downloading Clean JS"))

if __name__ == "__main__":
    collector = UltimateJSCollector()
    collector.run()