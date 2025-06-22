import argparse

import torch
import yaml
import time
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.spell_checker import NeuralTextCorrector


def load_configuration(configuration_path: str):
    with open(configuration_path) as config_stream:
        configuration_data = yaml.safe_load(config_stream)
    return configuration_data


def execute_correction(arguments):
    configuration = load_configuration(arguments.config_file)
    
    print("load model...")
    text_corrector = NeuralTextCorrector(model_name=configuration["model"],
                                        tokenizer_name=configuration["tokenizer"],
                                        vocabulary_size=configuration["n_words"],
                                        max_edit_distance=configuration["maximum_edit_distance"],
                                        min_length_by_distance=configuration["minimum_length_per_ed"],
                                        search_beam_size=configuration["beam_width"],
                                        error_penalties=configuration["penalties"],
                                        fix_valid_words=configuration["correct_real_words"],
                                        valid_word_cost=configuration["real_word_penalty"],
                                        initial_char_cost=configuration["first_char_penalty"],
                                        fix_whitespace=configuration["correct_spaces"],
                                        max_split_distance=configuration["maximum_edit_distance_splits"],
                                        filter_candidates=configuration["prune_candidates"],
                                        filter_paths=configuration["prune_beams"],
                                        filter_threshold=configuration["pruning_delta"])

    if configuration["input_file"] == "None" and not arguments.f:
        while True:
            user_input = input("> ")
            corrected_text = text_corrector.process_text(user_input, show_progress=configuration["verbose"])
            if not configuration["verbose"] and not configuration["out_file"]:
                print(corrected_text)

    else:
        source_file = arguments.f if arguments.f else configuration["input_file"]
        destination_file = arguments.o if arguments.o else configuration["output_file"]
        with open(source_file) as input_stream:
            text_lines = input_stream.read().splitlines()
        text_lines = text_lines[arguments.start:(arguments.end if arguments.end is None else arguments.end + 1)]
        if destination_file != "None":
            output_filename = destination_file
            if arguments.start is not None:
                output_filename += f".{arguments.start}-{arguments.end}"
            output_stream = open(output_filename, "w")
        else:
            output_stream = None
        total_processing_time = 0
        for text_line in tqdm(text_lines):
            is_all_uppercase = text_line.isupper()
            if is_all_uppercase:
                text_line = text_line.lower()
            processing_start = time.time()
            try:
                corrected_line = text_corrector.process_text(text_line, show_progress=configuration["verbose"])
            except RuntimeError:
                corrected_line = text_line
                print("WARNING! RuntimeError for sequence:", text_line)
            total_processing_time += time.time() - processing_start
            if is_all_uppercase:
                corrected_line = corrected_line.upper()
            if output_stream is not None:
                output_stream.write(corrected_line)
                output_stream.write("\n")
        if output_stream is not None:
            output_stream.close()


def main(argv=None):
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("config_file", type=str, default="config.yml", nargs="?")
    argument_parser.add_argument("--start", type=int, default=None, required=False)
    argument_parser.add_argument("--end", type=int, default=None, required=False)
    argument_parser.add_argument("-f", type=str, required=False)
    argument_parser.add_argument("-o", type=str, required=False)
    parsed_arguments = argument_parser.parse_args(argv)
    execute_correction(parsed_arguments)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())