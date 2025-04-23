import os
import pandas as pd
import pyarrow.parquet as pq
from logger import logger  # Assuming you have a logger module
from datasets import load_dataset
import kaggle
import zipfile
from pathlib import Path  # Use pathlib for cleaner path manipulation
import shutil
import re  # For cleaning Wikipedia XML data
import time

# Additional imports for data generation and scraping
import requests
from bs4 import BeautifulSoup  # For web scraping
import random
from tqdm import tqdm  # For progress bars
import uuid  # For generating unique IDs

# Additional imports for trade data
import yfinance as yf
# You may need to pip install these as well if you don't have them:
#   pip install beautifulsoup4 requests tqdm yfinance

class DatasetManager:
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)  # Store as Path object
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, component):
        """Load raw data for a component."""
        try:
            raw_path = self.raw_dir / f"{component}_dataset.csv"
            if raw_path.exists():
                data = pd.read_csv(raw_path)
                logger.info(f"Loaded raw data for {component}: {raw_path}")
                return data
            else:
                logger.warning(f"No raw data found for {component}")
                return None
        except Exception as e:
            logger.error(f"Error loading raw data for {component}: {e}")
            raise

    def load_processed_data(self, component):
        """Load processed data for a component."""
        try:
            processed_path = self.processed_dir / f"{component}_train.parquet"
            if processed_path.exists():
                data = pq.read_table(str(processed_path)).to_pandas()  # Convert Path to string
                logger.info(f"Loaded processed data for {component}: {processed_path}")
                return data
            else:
                logger.warning(f"No processed data found for {component}")
                return None
        except Exception as e:
            logger.error(f"Error loading processed data for {component}: {e}")
            raise

    def detect_new_datasets(self):
        """Detect new datasets in raw directory."""
        try:
            datasets = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv') or f.endswith('.parquet')]
            logger.info(f"Detected datasets: {datasets}")
            return datasets
        except Exception as e:
            logger.error(f"Error detecting new datasets: {e}")
            raise

    def preprocess_data(self, component):
        """Preprocess raw data and save to processed directory."""
        try:
            data = self.load_raw_data(component)
            if data is None:
                return

            # Example preprocessing (customize per component)
            data = data.dropna()
            processed_path = self.processed_dir / f"{component}_train.parquet"
            data.to_parquet(str(processed_path))  # Convert Path to string
            logger.info(f"Preprocessed data saved for {component}: {processed_path}")
        except Exception as e:
            logger.error(f"Error preprocessing data for {component}: {e}")
            raise

    def download_hf_dataset(self, dataset_name, split="train", cache_dir=None):
        """Downloads a dataset from Hugging Face Hub."""
        logger.info(f"Downloading Hugging Face dataset: {dataset_name} (split: {split})")
        try:
            if cache_dir is None:
                cache_dir = self.processed_dir / ".cache" # Store cache within processed dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure cache_dir is an absolute path string before passing
            absolute_cache_dir_str = str(cache_dir.resolve())
            logger.debug(f"Using cache directory (absolute string): {absolute_cache_dir_str}")
            headers = {
                'User-Agent': 'AthalaAdjutor/1.0 (contact@example.com)'
            }
            dataset = load_dataset(dataset_name, split=split, cache_dir=absolute_cache_dir_str)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to download/load Hugging Face dataset {dataset_name}: {e}")
            return None

    def download_kaggle_dataset(self, dataset_slug, file_name=None, unzip=True):
        """Downloads a dataset from Kaggle."""
        logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
        target_dir = self.processed_dir / "kaggle_downloads" / dataset_slug.replace('/', '_')
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            kaggle.api.authenticate() # Ensure authentication
            kaggle.api.dataset_download_files(dataset_slug, path=str(target_dir), quiet=False, unzip=False) # Download zip first
            
            downloaded_files = list(target_dir.glob('*.zip'))
            if not downloaded_files:
                logger.error(f"No zip file found after attempting download for {dataset_slug}")
                return None

            zip_path = downloaded_files[0] # Assume one zip file
            logger.info(f"Downloaded {zip_path.name} from Kaggle dataset: {dataset_slug}")

            if unzip:
                logger.info(f"Unzipping {zip_path.name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                logger.info(f"Unzipped files to {target_dir}")
                # Optionally remove the zip file after extraction
                # os.remove(zip_path) 
            
            # Return the directory containing the (potentially unzipped) files
            return target_dir
        except Exception as e:
            logger.error(f"Failed to download/unzip Kaggle dataset {dataset_slug}: {e}")
            # Attempt cleanup of potentially partial download
            shutil.rmtree(str(target_dir), ignore_errors=True)
            return None

    def prepare_dialog_data(self, dataset_name="daily_dialog", output_filename="dialog_train.parquet"):
        """Prepares dialog training data."""
        try:
            # Panggil dataset dengan benar
            dataset = self.download_hf_dataset(dataset_name, split="train")
            if dataset is None:
              return

            df = dataset.to_pandas()

            # Proses data
            if 'dialog' in df.columns:
                df['text'] = df['dialog'].apply(lambda x: "\n".join(x))
            elif 'text' in df.columns:
                df['text'] = df['text']
            else:
                logger.error("Kolom teks tidak ditemukan")
                return

            df[['text']].to_parquet(str(self.processed_dir / output_filename))  # Convert Path to string

        except Exception as e:
            logger.error(f"Gagal memproses dialog: {e}")

    def prepare_coding_data(self, dataset_name="codeparrot/github-code", output_filename="coding_train.parquet"):
        """Prepares coding training data."""
        try:
            dataset = self.download_hf_dataset(dataset_name, split="train")

            if dataset is None:
              return

            dataset = dataset.filter(lambda example: example['language'] == 'Python')
            df = dataset.to_pandas()
            df.rename(columns={'code': 'content'}, inplace=True)
            df[['content']].to_parquet(str(self.processed_dir / output_filename))  # Convert Path to string


        except Exception as e:
            logger.error(f"Gagal memproses kode: {e}")

    def prepare_math_data(self, dataset_name="gsm8k", output_filename="math_train.parquet"):
        """Prepares math training data."""
        try:

            dataset = self.download_hf_dataset(dataset_name, split="train", cache_dir = self.processed_dir / ".cache")
            if dataset is None:
              return

            df = dataset.to_pandas()
            df.rename(columns={'question': 'problem', 'answer': 'solution'}, inplace=True)
            df[['problem', 'solution']].to_parquet(str(self.processed_dir / output_filename))  # Convert Path to string


        except Exception as e:
            logger.error(f"Gagal memproses matematika: {e}")

    def prepare_network_data(self, dataset_slug="unsw-nb15", output_filename="network_train.parquet"):
        """Prepares network data using Kaggle API."""
        try:
            data_dir = self.download_kaggle_dataset(dataset_slug)
            if data_dir is None:
                return  # Exit if the dataset wasn't downloaded

            # Load data from all CSV files within the directory
            all_data = []
            for file_path in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    logger.info(f"Successfully loaded: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if not all_data:
                logger.error("No data loaded from CSV files.")
                return

            # Concatenate all DataFrames into a single DataFrame
            combined_data = pd.concat(all_data, ignore_index=True)

            # Save the combined data to a Parquet file
            parquet_path = self.processed_dir / output_filename
            combined_data.to_parquet(str(parquet_path))  # Convert Path to string
            logger.info(f"Combined network data saved to: {parquet_path}")

        except Exception as e:
            logger.error(f"Failed to process network data: {e}")

    def prepare_trading_data(self, output_filename="trading_train.parquet"):
         """Prepares trading data using yfinance."""
         try:
             logger.info("Preparing trading data using yfinance...")
             ticker = "BTC-USD"  # Bitcoin USD
             start_date = "2023-01-01"
             end_date = "2024-01-01"

             # Download data from yfinance
             data = yf.download(ticker, start=start_date, end=end_date)

             # Basic preprocessing (you can add more)
             data = data.dropna()

             # Add some technical indicators (example)
             data['SMA_20'] = data['Close'].rolling(window=20).mean()
             data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
             data['Volatility'] = data['High'] - data['Low']

             # Save to parquet
             parquet_path = self.processed_dir / output_filename
             data.to_parquet(str(parquet_path))
             logger.info(f"Trading data saved to: {parquet_path}")

         except Exception as e:
             logger.error(f"Error downloading or processing trading data: {e}")

    def prepare_captcha_data(self, dataset_slug="google-street-view-house-numbers", output_filename="captcha_train.parquet"):
        """Prepares CAPTCHA data from Kaggle."""
        try:
            data_dir = self.download_kaggle_dataset(dataset_slug)
            if data_dir is None:
                return

            # Find the train CSV file (the other one is extra)
            train_file_path = next((f for f in data_dir.glob("*.csv") if 'train' in f.name.lower()), None)

            if not train_file_path:
                logger.error("No 'train' CSV file found in the downloaded dataset.")
                return

            # Load data from the CSV
            try:
                df = pd.read_csv(train_file_path)
                logger.info(f"Successfully loaded: {train_file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {train_file_path}: {e}")
                return

            parquet_path = self.processed_dir / output_filename
            df.to_parquet(str(parquet_path))
            logger.info(f"CAPTCHA data saved to: {parquet_path}")

        except Exception as e:
            logger.error(f"Failed to process CAPTCHA data: {e}")

    def prepare_threat_intel_data(self, output_filename="threat_intel_train.parquet"):
        """Placeholder for preparing threat intelligence data."""
        logger.warning("Threat intelligence data preparation is a placeholder and not implemented.")
        # Replace this with actual threat intelligence data preparation logic.
        # May involve external APIs or local datasets.
        # Example (using a hypothetical API):
        # try:
        #     # Replace with your API key and endpoint
        #     api_key = "YOUR_THREAT_INTEL_API_KEY"
        #     endpoint = "https://api.example.com/threats"
        #     response = requests.get(endpoint, headers={"Authorization": f"Bearer {api_key}"})
        #     response.raise_for_status()
        #     data = response.json()
        #     df = pd.DataFrame(data) # Assuming the API returns JSON
        #     df.to_parquet(str(self.processed_dir / output_filename))
        # except Exception as e:
        #     logger.error(f"Error fetching threat intel data: {e}")
        pass

    def prepare_rag_data(self, output_filename="rag_train.parquet"):
        """Placeholder for preparing RAG data."""
        logger.warning("RAG data preparation is a placeholder and not implemented.")
        # This might involve combining text from other sources
        # Example:
        # 1. Load text data from various sources (e.g., text files, web pages)
        # 2. Chunk the text into smaller segments
        # 3. Embed the text chunks using a model like SentenceTransformers
        # 4. Store the embeddings in a vector database (e.g., FAISS)

        pass

    # --- New methods for advanced data sources ---

    def prepare_reddit_conversations(self, subreddit="CasualConversation", output_filename="reddit_conversations.parquet", start_year=2015, end_year=2023):
        """Extract Reddit conversations from Pushshift API."""
        try:
            logger.info(f"Extracting Reddit conversations from r/{subreddit} ({start_year}-{end_year})")
            all_posts = []
            for year in tqdm(range(start_year, end_year + 1), desc="Processing Years"):
                for month in range(1, 13):
                    #Pushshift API
                    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&after={year}-{str(month).zfill(2)}-01&before={year}-{str(month + 1).zfill(2)}-01&size=500"
                    try:
                        response = requests.get(url)
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        data = response.json()['data']
                        for post in data:
                            all_posts.append({
                                "title": post.get('title', ''),  # Title of the post
                                "selftext": post.get('selftext', ''), # Main content of the post
                                "author": post.get('author', ''), # Author of the post
                                "created_utc": post.get('created_utc', ''), # Time created
                                "url": post.get('url', ''), # URL of the post
                                "subreddit": post.get('subreddit', '') # Subreddit name
                            })
                        time.sleep(0.1) #sleep to avoid rate limit
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error during Pushshift request: {e}")

            df = pd.DataFrame(all_posts)
            df['text'] = df['title'] + '\n' + df['selftext']  # Combine title and text
            df[['text']].to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Reddit conversations saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error extracting Reddit conversations: {e}")

    def prepare_wikipedia_gutenberg(self, wikipedia_dump_url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2", gutenberg_dir="./data/gutenberg", output_filename="wikipedia_gutenberg.parquet"):
        """Combine Wikipedia dump with Project Gutenberg books."""
        try:
            logger.info("Combining Wikipedia dump and Project Gutenberg books.")

            # Wikipedia (Simplified Example - requires downloading and cleaning the dump)
            #  This is a *highly* simplified example. Processing the full Wikipedia XML dump is complex.
            #  This snippet just fetches a few articles directly.  You'd need to use a proper XML parser (e.g., `xml.etree.ElementTree`) and handle the bz2 decompression.
            #  For large-scale Wikipedia processing, consider using dedicated libraries like `wikiextractor`.
            wikipedia_articles = []
            # Fetch some example wikipedia pages
            example_titles = [
                # Technology
                "Artificial intelligence", "Machine learning", "Computer science",
                "Internet", "Digital technology", "Robotics",
                # History
                "World War II", "Ancient Egypt", "Roman Empire",
                "Industrial Revolution", "Renaissance", "French Revolution",
                "American Civil War", "Cold War", "Ancient Greece",
                # Science
                "Physics", "Chemistry", "Biology", 
                "Astronomy", "Evolution", "Quantum mechanics",
                # Arts & Culture
                "Classical music", "Renaissance art", "Modern art",
                "Literature", "Theatre", "Film history",
                # Philosophy
                "Western philosophy", "Eastern philosophy", "Ethics",
                # Social Sciences
                "Psychology", "Sociology", "Economics",
                "Political science", "Anthropology", "Archaeology",
                # Religion
                "Buddhism", "Christianity", "Islam",
                "Hinduism", "Judaism", "World religions",
                # Geography
                "Physical geography", "Human geography", "Climate",
                # Mathematics
                "Mathematics", "Geometry", "Algebra",
                # Medicine
                "Medicine", "Human anatomy", "Public health",
                # Other
                "Architecture", "Engineering", "Agriculture",
                "Education", "Sports", "Food science"
            ]
            for title in example_titles:
                try:
                    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={title}&prop=extracts&explaintext")
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    data = response.json()
                    page = next(iter(data['query']['pages'].values())) # Get the page content
                    wikipedia_articles.append({"title": title, "text": page['extract']}) # add page extract to text
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching Wikipedia article '{title}': {e}")
            wikipedia_df = pd.DataFrame(wikipedia_articles)
            wikipedia_df['source'] = "Wikipedia"

            # Gutenberg (Simplified Example)
            # This assumes you have downloaded Project Gutenberg books (e.g., as text files) into the `gutenberg_dir`.
            # You would likely need to do further cleaning and metadata extraction.
            gutenberg_texts = []
            if Path(gutenberg_dir).exists():
                for file_path in Path(gutenberg_dir).glob("*.txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                            text = f.read()
                            gutenberg_texts.append({"title": file_path.name, "text": text})
                    except Exception as e:
                        logger.warning(f"Error reading Gutenberg book {file_path}: {e}")
            gutenberg_df = pd.DataFrame(gutenberg_texts)
            gutenberg_df['source'] = "Gutenberg"

            # Combine and save
            combined_df = pd.concat([wikipedia_df, gutenberg_df], ignore_index=True)
            combined_df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Combined Wikipedia/Gutenberg data saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error combining Wikipedia/Gutenberg data: {e}")

    def prepare_stack_overflow(self, archive_url="https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z", output_filename="stackoverflow.parquet"):
        """Extract Stack Overflow data from Internet Archive.  This is a stub.
        Due to the size and complexity of the Stack Overflow archive, this requires more advanced processing.
        Consider using tools like 7-Zip to extract the XML files, then using an XML parser to extract the relevant data.
        This is left as a placeholder due to the complexity.
        """
        logger.warning("Stack Overflow data extraction is a placeholder and not implemented.")
        # This is a significantly more complex task.  Consider using:
        # 1.  `7z` command-line tool to extract the XML files.
        # 2.  `xml.etree.ElementTree` or `lxml` to parse the XML.
        # 3.  Iterate through the XML, extracting `Question` and `Answer` posts.
        # 4.  Clean the HTML content (Stack Overflow posts contain HTML).
        pass

    def prepare_github_codeparrot_plus(self, output_filename="github_codeparrot_plus.parquet", min_stars=100, start_year=2020, end_year=2025):
      """
      This method is a placeholder for a more complex process.
      It requires more advanced techniques for filtering GitHub repositories, parsing code,
      and handling large datasets.
      """
      logger.warning("GitHub CodeParrot+ data extraction is a placeholder and not implemented. Requires tree-sitter and GitHub API.")
      # This method would require significant work involving:
      # 1. Using the GitHub API to search for repositories with > min_stars and active commits in the specified years.
      # 2. Cloning the repositories.
      # 3. Using `tree-sitter` to parse the code and extract Abstract Syntax Trees (ASTs).
      # 4. Converting the ASTs into a suitable format for training.

    def prepare_lean4_theorem_proving(self, output_filename="lean4_theorems.parquet"):
        """Placeholder for extracting Lean4 theorem proving data. Requires knowledge of Lean4 format."""
        logger.warning("Lean4 theorem extraction is a placeholder and not implemented. Requires Lean4 parsing.")
        # Requires understanding the Lean4 language and file format.

    def prepare_imo_shortlists(self, output_filename="imo_problems.parquet", start_year=1959, end_year=2024):
        """
        Scrapes IMO problem shortlists from the official website.
        """
        try:
            logger.info(f"Scraping IMO problem shortlists ({start_year}-{end_year})")
            all_problems = []
            for year in tqdm(range(start_year, end_year + 1), desc="Processing Years"):
                url = f"https://www.imo-official.org/problems.aspx?yr={year}"
                try:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Adjust the selector based on the actual website structure. Inspect the website!
                    problem_tables = soup.find_all('table', class_='problem-table')  # Example class name

                    for table in problem_tables:
                        for row in table.find_all('tr'):
                            cells = row.find_all('td')
                            if len(cells) == 2: # Assuming each row has two cells (Problem and Solution/Answer)
                                problem = cells[0].text.strip()
                                #answer = cells[1].text.strip() # uncomment if solution is present and needs to be extracted
                                all_problems.append({"year": year, "problem": problem})  # , "solution": answer})

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching IMO problems for {year}: {e}")

            df = pd.DataFrame(all_problems)
            df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"IMO problems saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error extracting IMO problems: {e}")

    def prepare_cic_darknet2020(self, output_filename="cic_darknet2020.parquet"):
      """Placeholder for extracting CIC-Darknet2020 data.  Requires dataset download and parsing."""
      logger.warning("CIC-Darknet2020 extraction is a placeholder and not implemented. Requires dataset and parsing.")
      # This requires downloading the 2.5TB dataset from the University of New Brunswick.
      # It then requires parsing the network traffic data (likely pcap files).
      # Consider using tools like `tcpdump` or `Wireshark` to analyze the traffic.

    def prepare_malware_dna_db(self, output_filename="malware_dna.parquet"):
      """Placeholder for extracting Malware DNA data. Requires VirusShare/Hybrid Analysis access and API usage."""
      logger.warning("Malware DNA extraction is a placeholder and not implemented. Requires API access and malware analysis.")
      # This requires access to VirusShare or Hybrid Analysis and their APIs.
      # It then requires malware analysis to extract behavioral metadata.

    # --- Synthetic Data Generation Methods ---
    def generate_gan_captcha(self, num_captchas=1000000, output_filename="synthetic_captcha.parquet"):
      """Placeholder for GAN-based CAPTCHA generation. Requires GAN model."""
      logger.warning("GAN-based CAPTCHA generation is a placeholder and not implemented. Requires GAN model (StyleGAN3).")
      # This requires training a StyleGAN3 model on CAPTCHA images.
      # It then uses the trained model to generate synthetic CAPTCHAs.
      # The generated CAPTCHAs should have perturbations and noise.

    def simulate_darkweb_marketplace(self, num_transactions=10000, output_filename="darkweb_transactions.parquet"):
      """Placeholder for dark web marketplace simulation. Requires RL framework."""
      logger.warning("Dark web marketplace simulation is a placeholder and not implemented. Requires RL agents (Mesa framework).")
      # This requires building a simulation of a dark web marketplace.
      # It involves creating RL agents for sellers and buyers.
      # The simulation generates realistic transactions.

    def generate_bin_attack_data(self, num_records=100000, output_filename="bin_attack_data.parquet"):
        """Generates synthetic credit card data for BIN attacks."""
        try:
            logger.info(f"Generating {num_records} synthetic credit card records for BIN attacks.")

            # Sample BIN issuers (replace with a more comprehensive list)
            bin_issuers = ["411111", "522222", "377777"]

            data = []
            for _ in tqdm(range(num_records), desc="Generating Records"):
                bin_issuer = random.choice(bin_issuers)
                account_number = ''.join(random.choices('0123456789', k=16 - len(bin_issuer) - 1))  # Remaining digits (minus check digit)
                card_number = bin_issuer + account_number

                # Luhn Algorithm (Check Digit)
                sum_odd = 0
                sum_even = 0
                reverse_card_number = card_number[::-1]  # reversed string
                for i in range(len(reverse_card_number)):
                    digit = int(reverse_card_number[i])
                    if (i + 1) % 2 != 0:  # Odd position (1-based indexing)
                        sum_odd += digit
                    else:
                        doubled_digit = digit * 2
                        sum_even += doubled_digit - 9 if doubled_digit > 9 else doubled_digit  # Subtract 9 if doubled is > 9

                check_digit = (10 - ((sum_odd + sum_even) % 10)) % 10

                full_card_number = card_number + str(check_digit)
                expiry_month = str(random.randint(1, 12)).zfill(2)
                expiry_year = str(random.randint(2025, 2030))
                cvv = str(random.randint(100, 999))

                data.append({
                    "card_number": full_card_number,
                    "expiry_month": expiry_month,
                    "expiry_year": expiry_year,
                    "cvv": cvv
                })

            df = pd.DataFrame(data)
            df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Generated BIN attack data saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error generating BIN attack data: {e}")

    def generate_deepfake_financial_profiles(self, num_profiles=100, output_filename="deepfake_profiles.parquet"):
        """Menggunakan model lokal untuk generate data finansial"""
        try:
            logger.info(f"Generating {num_profiles} synthetic financial profiles using local models")
            
            # Ganti dengan model lokal (contoh: GPT-Neo)
            from transformers import pipeline
            generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
            
            # Template prompt
            prompt = """Generate a financial profile with:
            - Name: {name}
            - Address: {address}
            - Bank Balance: ${balance}
            - Credit Score: {score}
            """
            
            data = []
            for _ in range(num_profiles):
                # Generate data acak
                name = f"User_{uuid.uuid4().hex[:8]}"
                address = f"{random.randint(1,999)} Fake Street"
                balance = random.randint(1000, 1000000)
                score = random.randint(300, 850)
                
                # Generate text
                generated = generator(
                    prompt.format(name=name, address=address, balance=balance, score=score),
                    max_length=200,
                    do_sample=True,
                    temperature=0.7
                )
                
                data.append({
                    "name": name,
                    "address": address,
                    "balance": balance,
                    "score": score,
                    "document": generated[0]['generated_text']
                })
                
            df = pd.DataFrame(data)
            df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Generated financial profiles saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error generating financial profiles: {e}")

    # --- NEW FEATURE DATASETS (PLACEHOLDERS - IMPLEMENT ACCORDING TO NEEDS) ---
    def prepare_new_dialog_data(self, output_filename="new_dialog_data.parquet"):
        """Placeholder for a new dialog dataset source."""
        logger.warning("New Dialog data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape dialogs from a website
        # 2. Load dialog data from a local file
        # 3. Use an API to get dialog data
        pass  # Replace with actual data loading and processing

    def prepare_new_coding_data(self, output_filename="new_coding_data.parquet"):
        """Placeholder for a new coding dataset source."""
        logger.warning("New Coding data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape code snippets from a coding forum
        # 2. Load code from a GitHub repository
        # 3. Use an API to get code examples
        pass  # Replace with actual data loading and processing

    def prepare_new_math_data(self, output_filename="new_math_data.parquet"):
        """Placeholder for a new math dataset source."""
        logger.warning("New Math data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape math problems from a website
        # 2. Load math problems from a textbook
        # 3. Generate math problems using a symbolic math library
        pass  # Replace with actual data loading and processing

    def prepare_new_network_data(self, output_filename="new_network_data.parquet"):
        """Placeholder for a new network dataset source."""
        logger.warning("New Network data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Capture network traffic using tcpdump
        # 2. Load network traffic data from a pcap file
        # 3. Generate synthetic network traffic data
        pass  # Replace with actual data loading and processing

    def prepare_new_trading_data(self, output_filename="new_trading_data.parquet"):
        """Placeholder for a new trading dataset source."""
        logger.warning("New Trading data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Use a different trading API (e.g., Alpaca, IEX Cloud)
        # 2. Load trading data from a local CSV file
        # 3. Generate synthetic trading data
        pass  # Replace with actual data loading and processing

    def prepare_new_captcha_data(self, output_filename="new_captcha_data.parquet"):
        """Placeholder for a new CAPTCHA dataset source."""
        logger.warning("New CAPTCHA data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape CAPTCHAs from a website
        # 2. Generate CAPTCHAs using a CAPTCHA library
        # 3. Load CAPTCHAs from a local file
        pass  # Replace with actual data loading and processing

    def prepare_new_threat_intel_data(self, output_filename="new_threat_intel_data.parquet"):
        """Placeholder for a new threat intelligence dataset source."""
        logger.warning("New Threat Intel data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Use a different threat intelligence API
        # 2. Load threat intelligence data from a local file
        # 3. Scrape threat intelligence data from a website
        pass  # Replace with actual data loading and processing

    def prepare_new_rag_data(self, output_filename="new_rag_data.parquet"):
        """Placeholder for a new RAG dataset source."""
        logger.warning("New RAG data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Load data from a new knowledge base
        # 2. Scrape data from a website
        # 3. Create a new RAG pipeline
        pass  # Replace with actual data loading and processing

    def prepare_all_training_data(self):
        """Downloads and prepares all necessary training datasets."""
        logger.info("Starting preparation of all training datasets...")
        self.prepare_dialog_data()
        self.prepare_coding_data()
        self.prepare_math_data()
        self.prepare_network_data()
        self.prepare_trading_data()
        self.prepare_captcha_data()
        self.prepare_threat_intel_data()
        self.prepare_rag_data()

        # New methods:
        self.prepare_reddit_conversations()
        self.prepare_wikipedia_gutenberg()
        self.prepare_stack_overflow()
        self.prepare_github_codeparrot_plus()
        self.prepare_lean4_theorem_proving()
        self.prepare_imo_shortlists()
        self.prepare_cic_darknet2020()
        self.prepare_malware_dna_db()

        logger.info("Finished preparation of training datasets.")


if __name__ == "__main__":

    # Initialize and run the DatasetManager
    manager = DatasetManager()
    manager.prepare_all_training_data()