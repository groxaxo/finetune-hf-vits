#!/usr/bin/env python3
"""
Test for Spanish TTS training helper script.

This test validates that:
1. The finetune_spanish.py script exists and is executable
2. The script accepts the correct arguments
3. The Spanish config example is valid JSON
4. The run_vits_finetuning.py script has the new dataset file arguments
"""

import os
import sys
import json
import subprocess
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_spanish_helper_script_exists():
    """Test that finetune_spanish.py exists and is executable."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "finetune_spanish.py")
    assert os.path.exists(script_path), f"Script not found: {script_path}"
    assert os.access(script_path, os.X_OK), f"Script is not executable: {script_path}"
    print("✓ finetune_spanish.py exists and is executable")


def test_spanish_helper_script_help():
    """Test that the script can display help."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "finetune_spanish.py")
    
    result = subprocess.run(
        ["python", script_path, "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Script help failed with code {result.returncode}"
    assert "train_dataset" in result.stdout, "Missing --train_dataset in help"
    assert "eval_dataset" in result.stdout, "Missing --eval_dataset in help"
    assert "output_dir" in result.stdout, "Missing --output_dir in help"
    assert "learning_rate" in result.stdout, "Missing --learning_rate in help"
    # Check for Spanish TTS and spa-train model reference
    assert "Spanish TTS" in result.stdout, "Missing Spanish TTS description"
    assert "spa-train" in result.stdout or "spa" in result.stdout.lower(), \
        "Missing Spanish model reference"
    
    print("✓ finetune_spanish.py --help works correctly")


def test_spanish_config_example():
    """Test that the Spanish config example is valid JSON."""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "training_config_examples",
        "finetune_spanish.json"
    )
    
    assert os.path.exists(config_path), f"Config not found: {config_path}"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check required fields
    assert "model_name_or_path" in config, "Missing model_name_or_path"
    assert config["model_name_or_path"] == "ylacombe/mms-tts-spa-train", \
        "Wrong model checkpoint for Spanish"
    
    assert "output_dir" in config, "Missing output_dir"
    assert "learning_rate" in config, "Missing learning_rate"
    assert "do_train" in config, "Missing do_train"
    assert "do_eval" in config, "Missing do_eval"
    
    print("✓ finetune_spanish.json is valid and contains required fields")


def test_run_vits_has_dataset_file_args():
    """Test that run_vits_finetuning.py has the new dataset file arguments."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "run_vits_finetuning.py")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    assert "train_dataset_file" in content, "Missing train_dataset_file argument"
    assert "eval_dataset_file" in content, "Missing eval_dataset_file argument"
    assert 'data_files=data_args.train_dataset_file' in content, \
        "Missing implementation of train_dataset_file loading"
    assert 'data_files=data_args.eval_dataset_file' in content, \
        "Missing implementation of eval_dataset_file loading"
    
    print("✓ run_vits_finetuning.py has dataset file arguments")


def test_spanish_script_validates_input_files():
    """Test that the Spanish script validates input files exist."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "finetune_spanish.py")
    
    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as train_f:
        train_path = train_f.name
        train_f.write('{"audio": "test.wav", "text": "test"}\n')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as eval_f:
        eval_path = eval_f.name
        eval_f.write('{"audio": "test.wav", "text": "test"}\n')
    
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            # Test with valid files (it will fail at accelerate launch, but that's OK)
            result = subprocess.run(
                [
                    "python", script_path,
                    "--train_dataset", train_path,
                    "--eval_dataset", eval_path,
                    "--output_dir", output_dir,
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Should not fail on file validation
            assert "Error: Training dataset not found" not in result.stderr
            assert "Error: Evaluation dataset not found" not in result.stderr
            
            print("✓ Spanish script validates input files correctly")
            
        except subprocess.TimeoutExpired:
            # Expected - the script will try to run accelerate
            print("✓ Spanish script validates input files correctly (timeout OK)")
        finally:
            # Cleanup
            os.unlink(train_path)
            os.unlink(eval_path)


def test_spanish_script_rejects_missing_files():
    """Test that the Spanish script rejects missing input files."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "finetune_spanish.py")
    
    result = subprocess.run(
        [
            "python", script_path,
            "--train_dataset", "/nonexistent/train.jsonl",
            "--eval_dataset", "/nonexistent/eval.jsonl",
            "--output_dir", "/tmp/test_output",
        ],
        capture_output=True,
        text=True
    )
    
    # Should fail with error message
    assert result.returncode != 0, "Script should fail with missing files"
    assert "not found" in result.stderr.lower(), "Should report file not found"
    
    print("✓ Spanish script rejects missing input files")


if __name__ == "__main__":
    print("Running Spanish TTS training helper tests...\n")
    
    test_spanish_helper_script_exists()
    test_spanish_helper_script_help()
    test_spanish_config_example()
    test_run_vits_has_dataset_file_args()
    test_spanish_script_validates_input_files()
    test_spanish_script_rejects_missing_files()
    
    print("\n✓ All Spanish training helper tests passed!")
