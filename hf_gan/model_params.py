import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import json

def human_readable_number(number):
    if number >= 1000000000:
        return f"{number / 1000000000:.2f}B"
    elif number >= 1000000:
        return f"{number / 1000000:.2f}M"
    elif number >= 1000:
        return f"{number / 1000:.2f}K"
    else:
        return str(number)

def print_model_info(model_name, config_overrides=None, verbose=False, dump_config=False):
    """Loads model config, applies overrides, instantiates model, prints info."""
    try:
        config = AutoConfig.from_pretrained(model_name)

        if config_overrides:
            for key, value in config_overrides.items():
                try:
                    setattr(config, key, value)  # Apply config overrides
                except AttributeError:
                    print(f"Warning: Invalid config key '{key}'. Skipping.")

        if dump_config:
            for k,v in config.__dict__.items():
                print(f"{k}={v}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16, # Optional: For float16 dtype
            ignore_mismatched_sizes=True  
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


        if verbose:  # Print detailed parameter info if verbose is True
            for name, param in model.named_parameters():
                print(f"Parameter: {name}, Shape: {param.shape}")

        print(f"Total Parameters: {human_readable_number(total_params)}")
        print()
        print(f"Total Parameters: {total_params} ")
        print(f"Trainable Parameters:  {trainable_params}")

        del model  # Release memory

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print model info.")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name.",
        default='gpt2'
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed parameter information.",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store_true",
        help="Print detailed config information.",
    )
    # Add arguments for config overrides dynamically
    args, unknown_args = parser.parse_known_args() # Parse known and unknown args


    # Process additional keyword arguments for config overrides

    config_overrides = {}
    for arg in unknown_args:
        k,v=arg.split('=')
        print(f"Setting config {k}={v}")
        try: 
            try:
                value = json.loads(v) # Try to parse as JSON first
            except json.JSONDecodeError:
                pass # Keep as string if not valid JSON
            config_overrides[k] = value
        except IndexError:
            print(f"Error: Missing value for argument '{arg}'.")
            exit()


    print_model_info(args.model_name, config_overrides, args.verbose,args.config)