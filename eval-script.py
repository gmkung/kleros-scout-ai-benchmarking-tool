from typing import Any, Dict
import json
import os
from evals.eval import Eval
import requests
from dotenv import load_dotenv
from evals.api import CompletionFn
from evals.record import record_match


load_dotenv()


class ContractTaggingEval(Eval):
    def __init__(self, completion_fns: list[CompletionFn], samples_jsonl: str):
        super().__init__(completion_fns, samples_jsonl)
        self.api_key = os.getenv("API_KEY")

    def eval_sample(self, sample: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        # 1. Get API data
        contract_data = self._fetch_contract_data(sample["contract_address"])

        # 2. Create prompt
        prompt = self._create_prompt(sample, contract_data)

        # 3. Get model response
        response = self.completion_fn(prompt=prompt)

        # 4. Compare and record results
        result = self._compare(sample["expected"], response)
        correct = (
            result["accuracy"] >= 0.9
            and result["completeness"] == 1.0
            and result["formatting"] == 1.0
        )
        record_match(
            score=result["accuracy"],
            expected=sample["expected"],
            actual=response,
            metadata=result,
            correct=correct,
        )
        return result

    def _fetch_contract_data(self, contract_address: str) -> Dict[str, Any]:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"https://api.example.com/contracts/{contract_address}",
                headers=headers,
                timeout=10,  # Set a timeout of 10 seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            return {"error": f"HTTP error occurred: {str(e)}"}
        except requests.ConnectionError as e:
            return {"error": f"Connection error occurred: {str(e)}"}
        except requests.Timeout as e:
            return {"error": f"Timeout error occurred: {str(e)}"}
        except requests.RequestException as e:
            return {"error": f"Request error occurred: {str(e)}"}

    def _create_prompt(
        self, sample: Dict[str, Any], contract_data: Dict[str, Any]
    ) -> str:
        return f"""
        Contract Address: {sample['contract_address']}
        External Data: {json.dumps(contract_data)}
        
        Please tag this contract with the following information:
        - Project Name
        - Public Name Tag
        - UI Link
        """

    def _compare(self, expected: Dict[str, Any], actual: str) -> Dict[str, Any]:
        try:
            # Parse model output
            parsed_output = {}
            for line in actual.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    parsed_output[key] = value.strip()

            # Check completeness
            required_fields = ["project_name", "public_name_tag", "ui_link"]
            missing_fields = [
                field for field in required_fields if field not in parsed_output
            ]
            completeness_score = 1.0 if not missing_fields else 0.0

            # Check formatting
            formatting_errors = []
            if len(parsed_output.get("public_name_tag", "")) > 50:
                formatting_errors.append("Tag exceeds 50 characters")

            # Check accuracy
            accuracy_score = 1.0 if parsed_output == expected else 0.0

            return {
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "formatting": 1.0 if not formatting_errors else 0.0,
                "errors": formatting_errors,
            }
        except KeyError as e:
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "formatting": 0.0,
                "errors": [f"Key error: {str(e)}"],
            }
        except ValueError as e:
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "formatting": 0.0,
                "errors": [f"Value error: {str(e)}"],
            }
