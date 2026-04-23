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
        # ||example.com/ads.js 形式や /ads.js$script などを狙い撃ち
        found = re.findall(r"\|\|([a-z0-9.-]+\/.*\.js)", text)
        return [f"https://{p.split('$')[0]}" for p in found]

    def download_and_save(self, url, label):
        try:
            # $scriptオプションなどが付いている場合を考慮
            clean_url = url.split('$')[0]
            res = self.session.get(clean_url, timeout=5)
            if res.status_code == 200 and len(res.text) > 200 and not(res.text.startswith('<')):
                h = self.get_hash(res.text)
                if h not in self.hashes:
                    self.hashes.add(h)
                    with open(os.path.join(self.root, label, f"{h}.js"), "w", encoding="utf-8") as f:
                        f.write(res.text)
        except:
            pass

    def run(self):
        # 広告・トラッカーの収集
        sources = {
            "tracking": "https://malware-filter.gitlab.io/malware-filter/tracking-filter.txt",
            "easylist": "https://easylist.to/easylist/easylist.txt"
        }
        
        all_urls = []
        for name, url in sources.items():
            print(f"Fetching {name}...")
            raw = self.session.get(url).text
            all_urls.extend(self.extract_urls(raw))

        # 重複URLを排除してシャッフル
        all_urls = list(set(all_urls))
        print(f"Total unique URLs identified: {len(all_urls)}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(tqdm(executor.map(lambda u: self.download_and_save(u, "ad_tracker"), all_urls), 
                      total=len(all_urls), desc="Downloading Ad/Tracker JS"))

        # クリーンなデータの収集 (cdnjs API)
        print("Fetching clean libraries from cdnjs...")
        cdnjs_res = self.session.get("https://api.cdnjs.com/libraries?fields=latest").json()
        clean_urls = [lib['latest'] for lib in cdnjs_res['results'][:20000]]
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            list(tqdm(executor.map(lambda u: self.download_and_save(u, "clean"), clean_urls), 
                      total=len(clean_urls), desc="Downloading Clean JS"))

if __name__ == "__main__":
    collector = UltimateJSCollector()
    collector.run()