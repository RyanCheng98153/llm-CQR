import re
import csv
import json

def parse_log_to_csv(log_text, output_filename):
    # 1. Clean up the log: Remove the top header so Sample 0 is processed
    log_text = re.sub(r"Evaluation Log\n=+", "", log_text).strip()
    
    # 2. Split by the dashed separator
    samples = log_text.split('------------------------------')
    
    data_rows = []

    for sample in samples:
        sample = sample.strip()
        if not sample:
            continue
            
        # Extract fields using Regular Expressions
        # We use .strip() on results to clean up whitespace
        sample_id = re.search(r"Sample ID:\s*(\d+)", sample)
        status = re.search(r"Status:\s*(\w+)", sample)
        
        # History extraction: Capture everything between 'History:' and 'Original Query:'
        # Using a positive lookahead (?=...) ensures we don't consume the 'Original Query' label
        history_match = re.search(r"History:(.*?)(?=Original Query:)", sample, re.DOTALL)
        history_list = []
        if history_match:
            history_text = history_match.group(1).strip()
            # Split by lines, remove the bullet point '-' and clean whitespace
            history_list = [line.strip("- ").strip() for line in history_text.split('\n') if line.strip()]

        original_query = re.search(r"Original Query:\s*(.*)", sample)
        rewritten_query = re.search(r"Rewritten Query:\s*(.*)", sample)
        ground_truth = re.search(r"Ground Truth:\s*(.*)", sample)
        
        # Metrics extraction
        rouge_l = re.search(r"ROUGE-L:\s*([\d\.]+)", sample)
        bleu = re.search(r"BLEU:\s*([\d\.]+)", sample)
        latency = re.search(r"Latency:\s*([\d\.]+)", sample)

        # Only add to list if at least a Sample ID was found
        if sample_id:
            row = {
                "sample_id": sample_id.group(1),
                "status": status.group(1) if status else "",
                "history": json.dumps(history_list), # Save as a valid array-string
                "original_query": original_query.group(1).strip() if original_query else "",
                "rewritten_query": rewritten_query.group(1).strip() if rewritten_query else "",
                "groundtruth": ground_truth.group(1).strip() if ground_truth else "",
                "Latency": latency.group(1) if latency else "",
                "RougeL": rouge_l.group(1) if rouge_l else "",
                "Bleu": bleu.group(1) if bleu else "",
                "Human": "" # Left empty as requested
            }
            data_rows.append(row)

    # Define CSV column headers
    fieldnames = ["sample_id", "status", "history", "original_query", "rewritten_query", "groundtruth", "Latency", "RougeL", "Bleu", "Human"]

    # Write to CSV
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)

    print(f"Successfully converted {len(data_rows)} samples to {output_filename}")

if __name__ == "__main__":
    with open("qrecc_results_detailed.txt", "r", encoding="utf-8") as f:
        log_content = f.read()
    
    parse_log_to_csv(log_content, "qrecc_results_detailed.csv")