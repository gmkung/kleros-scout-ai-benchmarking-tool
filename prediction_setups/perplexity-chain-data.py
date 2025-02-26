import json
import requests
import re
from typing import Dict, List, Optional, Any
import time
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()


def get_chain_name(rich_address: str) -> str:
    """Get chain name based on chain ID."""
    chain_map = {
        1: "Ethereum",
        10: "Optimism",
        56: "BNB Smart Chain",
        137: "Polygon",
        42161: "Arbitrum",
        43114: "Avalanche",
    }
    match = re.match(r"eip155:(\d+):", rich_address)
    if not match:
        return "Unknown"
    chain_id = int(match.group(1))
    return chain_map.get(chain_id, "Unknown")


def get_explorer_domain_to_block(rich_address: str) -> List[str]:
    """Get the specific explorer domain to block based on chain ID."""
    match = re.match(r"eip155:(\d+):", rich_address)
    if not match:
        return []

    chain_id = int(match.group(1))
    explorer_map = {
        1: "-etherscan.io",
        10: "-optimistic.etherscan.io",
        56: "-bscscan.com",
        100: "-gnosisscan.io",
        137: "-polygonscan.com",
        8453: "-basescan.org",
        42161: "-arbiscan.io",
        43114: "-snowscan.xyz",
        250: "-ftmscan.com",
        324: "-era.zksync.network",
        1285: "-moonriver.moonscan.io",
    }

    domain = explorer_map.get(chain_id)
    return [domain] if domain else []


def query_perplexity(address: str, chain_name: str) -> Optional[Dict[str, str]]:
    """Query Perplexity API for token information."""
    url = "https://api.perplexity.ai/chat/completions"
    api_key = os.getenv("PERPLEXITY_API_KEY")

    if not api_key:
        logging.error("PERPLEXITY_API_KEY not found in environment variables")
        return None

    prompt = f"""Given this smart contract address on {chain_name}: {address}
    Please return information about this smart contract in this exact JSON format:
    {{
        "Project Name": "project name",
        "Public Name Tag": "name/label of this smart contract ",
        "UI/Website Link": "domain of the project website ",
        "Public Note": "brief description of the purpose of this contract in the context of the project"
    }}
    Be concise in the description. If you're not certain about any field, make a best gues. All fields with no found information must be left as an empty string. 
    Return just the JSON and nothing else."""

    payload = {
        "model": "sonar",  # Updated to correct model name
        "messages": [
            {
                "role": "system",
                "content": "You are a blockchain information expert. Provide accurate, concise information about the contracts.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 250,
        "temperature": 0.1,
        "search_domain_filter": get_explorer_domain_to_block(address),
        "return_related_questions": False,
        "return_images": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logging.info(f"Sending request for {address}")
        logging.info(f"Request payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload, headers=headers)
        logging.info(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            logging.error(f"API error: {response.text}")
            return None

        result = response.json()
        logging.info(f"Raw API response: {result}")

        # Parse the JSON string from the response content
        content = result["choices"][0]["message"]["content"]
        logging.info(f"Content to parse: {content}")

        # Clean up the content by removing markdown code blocks
        cleaned_content = content.strip()
        if cleaned_content.startswith("```"):
            # Remove first line and last line if they contain ```
            lines = cleaned_content.split("\n")
            cleaned_content = "\n".join(lines[1:-1])

        # Remove any remaining "json" marker
        cleaned_content = cleaned_content.replace("json", "").strip()

        logging.info(f"Cleaned content: {cleaned_content}")
        data = json.loads(cleaned_content)
        return data

    except Exception as e:
        logging.error(f"Error querying Perplexity API: {str(e)}")
        return None


def validate_prediction(data: Dict[str, str]) -> bool:
    """Validate that all required fields are present and non-empty."""
    required_fields = [
        "Project Name",
        "Public Name Tag",
        "UI/Website Link",
        "Public Note",
    ]

    try:
        # Check all required fields exist and are non-empty strings
        for field in required_fields:
            if (
                not data.get(field)
                or not isinstance(data[field], str)
                or not data[field].strip()
            ):
                logging.warning(f"Missing or invalid {field}")
                return False
        return True
    except Exception as e:
        logging.warning(f"Validation error: {str(e)}")
        return False


def main():
    logging.info("Starting script...")

    try:
        # Update paths to be relative to script location
        input_path = "data/ground-truth/data-set1-mini.jsonl"

        # Create output directory
        os.makedirs("data/predictions", exist_ok=True)

        # Dynamically create output path by adding _result before the extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"data/predictions/{base_name}_result.jsonl"

        with open(input_path, "r") as f:
            entries = [json.loads(line) for line in f]

        logging.info(f"Loaded {len(entries)} entries from ground truth")

        # Initialize predictions list with same length as ground truth
        predictions = []
        failed_addresses = []

        # Process each entry and write immediately to maintain order
        with open(output_path, "w") as f:
            for entry in entries:
                address = entry["Contract Address"]
                chain = get_chain_name(address)

                logging.info(f"Processing {address} on {chain}")
                result = query_perplexity(address, chain)

                # Create prediction entry, using empty values if API call fails
                prediction = {
                    "Contract Address": address,
                    "Project Name": "",
                    "Public Name Tag": "",
                    "UI/Website Link": "",
                    "Public Note": "",
                }

                if result and validate_prediction(result):
                    # Update prediction with API results
                    prediction.update(result)
                    logging.info(f"Successfully got prediction for {address}")
                else:
                    failed_addresses.append(address)
                    logging.warning(f"No valid prediction obtained for {address}")

                # Write the prediction (complete or empty) to maintain line matching
                f.write(json.dumps(prediction) + "\n")
                predictions.append(prediction)

                time.sleep(1)

        logging.info(f"Processed {len(entries)} entries")
        logging.info(
            f"Got {len(entries) - len(failed_addresses)} successful predictions"
        )
        logging.info(f"Failed to get predictions for {len(failed_addresses)} addresses")

        if failed_addresses:
            logging.info("Failed addresses:")
            for addr in failed_addresses:
                logging.info(f"  - {addr}")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
