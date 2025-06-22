import os.path
import pickle
from typing import Set, Dict, Tuple, List, Optional

from itertools import combinations
from tqdm import tqdm
from nltk import edit_distance


PREDEFINED_TERMS = [
    "I'm",
    "can't",
    "Can't",
    "don't",
    "Don't",
    "won't",
    "Won't",
    "it's",
    "It's",
    "he's",
    "He's",
    "she's",
    "She's",
    "we'll",
    "We'll",
    "you'll",
    "You'll",
    "he'd",
    "He'd",
    "she'd",
    "She'd",
    "we'd",
    "We'd",
    "they'd",
    "They'd"
]


def create_partial_text(term: str, positions_to_skip: Set[int]) -> str:
    character_list = []
    for character_position, character in enumerate(term):
        if character_position not in positions_to_skip:
            character_list.append(character)
    return "".join(character_list)


def generate_partial_variations(term: str, characters_to_skip: int) -> Set[str]:
    partial_variations = set()
    character_positions = list(range(len(term)))
    for skip_positions in combinations(character_positions, characters_to_skip):
        skip_positions_set = set(skip_positions)
        partial_text = create_partial_text(term, skip_positions_set)
        partial_variations.add(partial_text)
    return partial_variations


class TextCandidateEngine:
    def __init__(self,
                 vocabulary_size: int,
                 max_distance: int,
                 min_lengths: Dict[int, int],
                 enable_space_fixes: bool,
                 max_splits: int):
        self.vocabulary_size = vocabulary_size
        self.valid_terms = set()
        self._load_vocabulary()
        self._include_predefined_terms()
        self.max_distance = max_distance
        self.min_lengths = min_lengths
        self.partial_term_mapping = {}
        self._initialize_partial_mapping()
        self.enable_space_fixes = enable_space_fixes
        self.max_splits = max_splits

    def _load_vocabulary(self):
        for text_line in open("data/word_frequencies.txt"):
            vocabulary_term = text_line.split()[0]
            term_is_valid = True
            for character in vocabulary_term:
                if not character.isalpha() and character != "-":
                    term_is_valid = False
                    break
            if not term_is_valid:
                continue
            self.valid_terms.add(vocabulary_term)
            if len(self.valid_terms) == self.vocabulary_size:
                break

    def _include_predefined_terms(self):
        for predefined_term in PREDEFINED_TERMS:
            self.valid_terms.add(predefined_term)

    def _build_partial_mapping(self):
        print("creating word stump index...")
        for vocabulary_term in tqdm(self.valid_terms):
            if vocabulary_term not in self.partial_term_mapping:
                self.partial_term_mapping[vocabulary_term] = set()
            self.partial_term_mapping[vocabulary_term].add(vocabulary_term)
            for distance_value in range(1, self.max_distance + 1):
                if len(vocabulary_term) < self.min_lengths[distance_value]:
                    continue
                partial_variations = generate_partial_variations(vocabulary_term, distance_value)
                for partial_variation in partial_variations:
                    if partial_variation not in self.partial_term_mapping:
                        self.partial_term_mapping[partial_variation] = set()
                    self.partial_term_mapping[partial_variation].add(vocabulary_term)

    def _initialize_partial_mapping(self):
        cache_location = "data/word_stump_index.pkl"
        if os.path.exists(cache_location):
            with open(cache_location, "rb") as cache_file:
                self.partial_term_mapping = pickle.load(cache_file)
        else:
            self._build_partial_mapping()
            with open(cache_location, "wb") as cache_file:
                pickle.dump(self.partial_term_mapping, cache_file)

    def _search_partial_mapping(self, term) -> Set[str]:
        potential_matches = set()
        potential_matches.add(term)
        if term in self.partial_term_mapping:
            potential_matches.update(self.partial_term_mapping[term])
        for skip_count in range(1, min(len(term), self.max_distance) + 1):
            partial_variations = generate_partial_variations(term, skip_count)
            for partial_variation in partial_variations:
                if partial_variation in self.partial_term_mapping:
                    potential_matches.update(self.partial_term_mapping[partial_variation])
        return potential_matches

    def _apply_match_filters(self,
                           term: str,
                           potential_matches: Set[str],
                           distance_limit: int,
                           check_initial_character: bool) \
            -> List[Tuple[str, int, bool]]:
        filtered_matches = []
        for potential_match in potential_matches:
            if term == potential_match:
                continue
            if check_initial_character and term[0].lower() != potential_match[0].lower():
                continue
            calculated_distance = edit_distance(term, potential_match, transpositions=True)
            if calculated_distance <= distance_limit and len(potential_match) >= self.min_lengths[calculated_distance]:
                filtered_matches.append((potential_match, calculated_distance, False))
        return filtered_matches

    def _find_term_alternatives(self, term: str, distance_limit: int, check_initial_character=False) \
            -> List[Tuple[str, int, bool]]:
        potential_matches = self._search_partial_mapping(term)
        filtered_matches = self._apply_match_filters(term, potential_matches, distance_limit, check_initial_character)
        return filtered_matches

    def _find_split_alternatives(self, input_token: str, distance_limit: int) -> List[Tuple[str, int, bool]]:
        split_alternatives = []
        for split_position in range(1, len(input_token)):
            left_part = input_token[:split_position]
            left_alternatives = []
            if self.check_valid_word(left_part):
                left_alternatives.append((left_part, 0, False))
            left_alternatives.extend(self._find_term_alternatives(left_part, distance_limit=distance_limit - 1))
            right_part = input_token[split_position:]
            right_alternatives = []
            if self.check_valid_word(right_part):
                right_alternatives.append((right_part, 0, False))
            right_alternatives.extend(self._find_term_alternatives(right_part, distance_limit=distance_limit - 1))
            for left_option, left_distance, _ in left_alternatives:
                for right_option, right_distance, _ in right_alternatives:
                    if left_distance + right_distance + 1 <= distance_limit:
                        split_alternatives.append((left_option + " " + right_option, left_distance + right_distance + 1, False))
        return split_alternatives

    def _find_merge_alternatives(self, first_token: str, second_token: str) -> List[Tuple[str, int, bool]]:
        combined_token = first_token + second_token
        merge_alternatives = []
        if self.check_valid_word(combined_token):
            merge_alternatives.append((combined_token, 1, True))
        term_alternatives = self._find_term_alternatives(combined_token, distance_limit=self.max_distance - 1)
        for alternative_term, distance_value, _ in term_alternatives:
            merge_alternatives.append((alternative_term, distance_value + 1, True))
        return merge_alternatives

    def _collect_all_alternatives(self, input_token: str, subsequent_token: Optional[str]) -> List[Tuple[str, int, bool]]:
        all_alternatives = self._find_term_alternatives(input_token, distance_limit=self.max_distance)
        if self.enable_space_fixes:
            all_alternatives.extend(self._find_split_alternatives(input_token, distance_limit=self.max_splits))
            if subsequent_token is not None and subsequent_token.isalpha():
                all_alternatives.extend(self._find_merge_alternatives(input_token, subsequent_token))
        return all_alternatives

    def generate_alternatives(self, input_token: str, subsequent_token: Optional[str]) -> List[Tuple[str, int, bool]]:
        all_alternatives = self._collect_all_alternatives(input_token, subsequent_token)
        if input_token[0].isupper() and len(input_token) == 1 or input_token[1:].islower():
            lowercase_alternatives = self._collect_all_alternatives(input_token.lower(), subsequent_token)
            processed_lowercase = []
            for alternative_term, distance_value, delay_flag in lowercase_alternatives:
                capitalized_alternative = alternative_term[0].upper() + alternative_term[1:]
                if self.check_valid_word(capitalized_alternative):
                    continue
                if input_token == capitalized_alternative:
                    continue
                processed_lowercase.append((capitalized_alternative, distance_value, delay_flag))
            all_alternatives = list(set(all_alternatives).union(set(processed_lowercase)))
        return all_alternatives

    def check_valid_word(self, term):
        return term in self.valid_terms