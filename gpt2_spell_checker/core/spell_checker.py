from typing import Dict

import torch.cuda
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import numpy as np
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from candidate_generator import TextCandidateEngine


def initialize_language_model(model_identifier: str) -> GPT2LMHeadModel:
    language_model = GPT2LMHeadModel.from_pretrained(model_identifier)
    language_model.eval()
    return language_model


def initialize_text_tokenizer(tokenizer_identifier) -> GPT2Tokenizer:
    text_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_identifier)
    return text_tokenizer


def split_into_tokens(input_text):
    return re.findall(r"\w[-'\w]*|.", input_text)


class NeuralTextCorrector:
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 vocabulary_size: int,
                 max_edit_distance: int,
                 min_length_by_distance: Dict[int, int],
                 search_beam_size: int,
                 error_penalties: Dict[int, float],
                 fix_valid_words: bool,
                 valid_word_cost: float,
                 initial_char_cost: float,
                 fix_whitespace: bool,
                 max_split_distance: int,
                 filter_candidates: bool,
                 filter_paths: bool,
                 filter_threshold: float):
        self.language_model = initialize_language_model(model_name)
        self.text_tokenizer = initialize_text_tokenizer(tokenizer_name)
        self.candidate_engine = TextCandidateEngine(vocabulary_size=vocabulary_size,
                                                   max_distance=max_edit_distance,
                                                   min_lengths=min_length_by_distance,
                                                   enable_space_fixes=fix_whitespace,
                                                   max_splits=max_split_distance)
        self.search_beam_size = search_beam_size
        self.error_penalties = error_penalties
        self.fix_valid_words = fix_valid_words
        self.valid_word_cost = valid_word_cost
        self.initial_char_cost = initial_char_cost
        self.filter_candidates = filter_candidates
        self.filter_paths = filter_paths
        self.filter_threshold = filter_threshold
        self.processing_device = "cpu"
        self._configure_processing_device()
        self.candidate_generation_time = 0

    def _configure_processing_device(self):
        if torch.cuda.is_available():
            self.processing_device = "cuda"
            self.language_model = self.language_model.to(self.processing_device)

    def _create_initial_path(self):
        initial_input = self.text_tokenizer(self.text_tokenizer.bos_token, return_tensors="pt")
        initial_input = initial_input.input_ids.to(self.processing_device)
        with torch.no_grad():
            model_output = self.language_model(initial_input)
        path_state = {
            "total_cost": 0,
            "probability_distribution": torch.softmax(model_output.logits[0][0], dim=0),
            "cached_states": model_output.past_key_values,
            "generated_text": "",
            "pending_steps": 0,
        }
        return path_state

    def _refresh_path_states(self, path_collection):
        for path_state in path_collection:
            if "probability_distribution" in path_state:
                continue
            token_inputs = self.text_tokenizer(path_state["proposed_text"], return_tensors="pt")["input_ids"].to(self.processing_device)
            with torch.no_grad():
                model_result = self.language_model(token_inputs, past_key_values=path_state["cached_states"])
            path_state["probability_distribution"] = torch.softmax(model_result.logits[0][0], dim=0)
            path_state["cached_states"] = model_result.past_key_values
        return path_collection

    def _convert_candidates_to_tokens(self, text_candidates):
        token_sequences = []
        for candidate_text in text_candidates:
            token_sequence = self.text_tokenizer(candidate_text)["input_ids"]
            token_sequences.append(token_sequence)
        return token_sequences

    def _generate_correction_options(self, current_token, following_token, has_preceding_space):
        correction_options = [(current_token, 0, False)]
        if (not self.candidate_engine.check_valid_word(current_token) or self.fix_valid_words) and current_token[0].isalpha():
            correction_options.extend(self.candidate_engine.generate_alternatives(current_token, following_token))
        if has_preceding_space:
            correction_options = [(" " + option_text, distance, is_split) for option_text, distance, is_split in correction_options]
        return correction_options

    def _find_optimal_single_token_cost(self, path_collection, original_token, correction_options, tokenized_options):
        minimum_cost = np.inf
        for path_state in path_collection:
            log_probabilities = torch.log(path_state["probability_distribution"])
            for option_index in range(len(correction_options)):
                token_sequence = tokenized_options[option_index]
                if len(token_sequence) == 1:
                    token_log_prob = log_probabilities[token_sequence[0]]
                    edit_distance = correction_options[option_index][1]
                    path_cost = self._calculate_path_cost(path_state["total_cost"], token_log_prob, original_token, correction_options[option_index][0], edit_distance)
                    if path_cost < minimum_cost:
                        minimum_cost = path_cost
        return minimum_cost

    def _calculate_path_cost(self, current_cost, log_probability, original_token, corrected_token, edit_distance):
        updated_cost = current_cost - log_probability
        updated_cost += self.error_penalties[edit_distance]
        if edit_distance > 0:
            if self.candidate_engine.check_valid_word(original_token):
                updated_cost += self.valid_word_cost
            initial_character = corrected_token[0] if corrected_token[0] != " " else corrected_token[1]
            if original_token[0] != initial_character:
                updated_cost += self.initial_char_cost
        return updated_cost

    def _choose_optimal_paths(self, path_collection):
        pending_paths = []
        active_paths = []
        for path_state in path_collection:
            if path_state["pending_steps"] > 0:
                pending_paths.append(path_state)
            else:
                active_paths.append(path_state)
        active_paths = sorted(active_paths, key=lambda p: p["total_cost"])
        pending_paths = sorted(pending_paths, key=lambda p: p["total_cost"])
        return active_paths[:self.search_beam_size] + pending_paths[:self.search_beam_size]

    def _filter_suboptimal_paths(self, path_collection):
        active_costs = [np.inf]
        pending_costs = [np.inf]
        for path_state in path_collection:
            path_cost = path_state["total_cost"]
            if path_state["pending_steps"] > 0:
                pending_costs.append(path_cost)
            else:
                active_costs.append(path_cost)
        active_cutoff = min(active_costs) + self.filter_threshold
        pending_cutoff = min(pending_costs) + self.filter_threshold
        filtered_paths = []
        for path_state in path_collection:
            if (path_state["pending_steps"] > 0 and path_state["total_cost"] < pending_cutoff) \
                    or (path_state["pending_steps"] == 0 and path_state["total_cost"] < active_cutoff):
                filtered_paths.append(path_state)
        return filtered_paths

    def _execute_search_iteration(self, path_collection, current_token, following_token, has_preceding_space, show_progress):
        model_invocations = 0
        generation_start = time.time()
        correction_options = self._generate_correction_options(current_token, following_token, has_preceding_space)
        generation_duration = time.time() - generation_start
        self.candidate_generation_time += generation_duration
        if show_progress:
            print(f"{len(correction_options)} candidates ({generation_duration:.4f} seconds)")
        tokenized_options = self._convert_candidates_to_tokens([option[0] for option in correction_options])
        updated_paths = []
        iteration_best_cost = self._find_optimal_single_token_cost(path_collection, current_token, correction_options, tokenized_options)
        
        for path_state in path_collection:
            if path_state["pending_steps"] > 0:
                path_state["pending_steps"] -= 1
                updated_paths.append(path_state)
                continue
            for option_idx, (corrected_text, edit_distance, requires_split) in enumerate(correction_options):
                token_sequence = tokenized_options[option_idx]
                initial_probability = path_state["probability_distribution"][token_sequence[0]]
                log_probability = torch.log(initial_probability)
                if len(token_sequence) > 1:
                    if self.filter_candidates:
                        single_token_cost = self._calculate_path_cost(path_state["total_cost"], log_probability, current_token, corrected_text, edit_distance)
                        if single_token_cost > iteration_best_cost + self.filter_threshold:
                            continue
                    token_tensor = self.text_tokenizer(corrected_text, return_tensors="pt")["input_ids"].to(self.processing_device)
                    with torch.no_grad():
                        model_output = self.language_model(token_tensor[:, :-1], past_key_values=path_state["cached_states"])
                    token_probabilities = torch.softmax(model_output["logits"][0], dim=1)
                    for position, token_id in enumerate(token_tensor[0][1:]):
                        log_probability += torch.log(token_probabilities[position, token_id])
                    model_invocations += 1
                path_cost = self._calculate_path_cost(path_state["total_cost"], log_probability, current_token, corrected_text, edit_distance)
                iteration_best_cost = min(path_cost, iteration_best_cost)
                new_path_state = {
                    "total_cost": path_cost,
                    "generated_text": path_state["generated_text"] + corrected_text,
                    "cached_states": path_state["cached_states"],
                    "proposed_text": corrected_text,
                    "pending_steps": 1 if requires_split else 0
                }
                updated_paths.append(new_path_state)
        updated_paths = self._choose_optimal_paths(updated_paths)
        if self.filter_paths:
            updated_paths = self._filter_suboptimal_paths(updated_paths)
        path_collection = self._refresh_path_states(updated_paths)
        model_invocations += len(path_collection)
        if show_progress:
            for path_state in path_collection:
                current_cost = path_state["total_cost"].item()
                current_text = path_state["generated_text"]
                print(f"{current_cost:.4f} {current_text}")
        return path_collection

    def process_text(self, input_text, show_progress=False):
        processing_start = time.time()
        self.candidate_generation_time = 0

        text_tokens = split_into_tokens(input_text)

        initial_path = self._create_initial_path()
        path_collection = [initial_path]

        has_preceding_space = False
        processed_word_count = 0
        for token_index, current_token in enumerate(text_tokens):
            is_dictionary_word = self.candidate_engine.check_valid_word(current_token)
            if show_progress and current_token != " ":
                processed_word_count += 1
                print(f"== step {processed_word_count} ==")
                print(f"token: {current_token} (real word: {is_dictionary_word})")
            iteration_start = time.time()
            if current_token == " ":
                has_preceding_space = True
            else:
                if token_index + 2 < len(text_tokens) and text_tokens[token_index + 1] == " ":
                    following_token = text_tokens[token_index + 2]
                else:
                    following_token = None
                path_collection = self._execute_search_iteration(path_collection, current_token, following_token, has_preceding_space, show_progress)
                has_preceding_space = False
            iteration_duration = time.time() - iteration_start

        if show_progress:
            total_duration = time.time() - processing_start
            print("== result ==")
            print(path_collection[0]["generated_text"])
            print(f"{total_duration:.4f} seconds ({self.candidate_generation_time:.4f} seconds for candidate generation)")

        return path_collection[0]["generated_text"]