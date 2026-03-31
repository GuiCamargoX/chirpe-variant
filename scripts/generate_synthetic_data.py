#!/usr/bin/env python3
"""Generate synthetic PSYCHS interview data for testing."""

import argparse
import json
import random
from pathlib import Path

import numpy as np


# Sample PSYCHS-like questions
PSYCHS_QUESTIONS = {
    "P1_Unusual_Thoughts": [
        "Have you ever had the feeling that something odd is going on?",
        "Have you ever been confused whether something was real or imaginary?",
        "Have you ever felt that events had special meaning for you?",
        "Do you ever feel like the radio or TV is communicating directly with you?",
        "Have you ever felt that people can read your mind?",
    ],
    "P2_Suspiciousness": [
        "Have you ever felt like people are talking about you?",
        "Do you ever feel suspicious of other people?",
        "Have you ever felt like you are being watched?",
        "Do you feel you have to pay close attention to feel safe?",
        "Have you felt someone might be trying to hurt you?",
    ],
    "P3_Unusual_Somatic": [
        "Have you worried something might be wrong with your body?",
        "Have you noticed your body seems different to others?",
        "Have you worried about your body shape?",
        "Do you feel something odd is happening with your body?",
    ],
    "P4_Disoriented": [
        "Do you ever feel confused about where you are?",
        "Do you lose track of time frequently?",
        "Do you have trouble concentrating?",
        "Do you find it hard to focus your attention?",
    ],
    "P5_Focused_Thoughts": [
        "Do your thoughts ever race quickly?",
        "Do you find your mind jumping between ideas?",
        "Do you get easily distracted?",
        "Do you find it hard to follow one train of thought?",
    ],
    "P6_Experiences": [
        "Have you ever heard voices others couldn't hear?",
        "Have you seen things that weren't really there?",
        "Do you have unusual sensory experiences?",
        "Have you noticed things others don't seem to perceive?",
    ],
    "P7_Perceiving": [
        "Do you hear noises or sounds that bother you?",
        "Do you notice unusual sensations in your body?",
        "Are you sensitive to certain sounds or lights?",
    ],
    "P8_Confusion": [
        "Do you often feel uncertain about things?",
        "Do you feel puzzled or confused easily?",
        "Do you find yourself feeling perplexed?",
    ],
    "P9_Daydreaming": [
        "Do you daydream a lot?",
        "Do you get lost in your own thoughts?",
        "Do you have an active imagination?",
        "Do you fantasize frequently?",
    ],
    "P10_Time_Experience": [
        "Does time ever seem to move strangely for you?",
        "Have you felt time was moving too fast or slow?",
        "Does time sometimes feel unreal to you?",
    ],
    "P11_Depersonalisation": [
        "Do you ever feel like you're not really here?",
        "Do you feel disconnected from yourself?",
        "Have you felt like you're watching yourself from outside?",
    ],
    "P12_Derealisation": [
        "Does the world sometimes feel unreal to you?",
        "Have you felt like you're in a dream?",
        "Does reality sometimes seem strange?",
    ],
    "P13_Superstitious": [
        "Are you superstitious?",
        "Do you believe in omens or signs?",
        "Do you feel you have a sixth sense?",
    ],
    "P14_Control": [
        "Have you felt forces outside yourself controlling you?",
        "Do you feel your thoughts are interfered with?",
        "Have you felt something putting thoughts in your head?",
    ],
    "P15_Reading_Minds": [
        "Have you felt people could read your mind?",
        "Do you think others know what you're thinking?",
        "Have you felt your thoughts are being broadcast?",
    ],
}

# CHR-P indicator phrases
CHR_INDICATORS = [
    "Yes, sometimes I feel that way",
    "I have experienced that",
    "It happens occasionally",
    "I notice that quite often",
    "That sounds familiar to me",
    "I have had that experience",
    "It occurs from time to time",
    "I can relate to that feeling",
    "That describes my experience",
    "I have felt that way before",
]

# Control/non-CHR responses
CONTROL_RESPONSES = [
    "No, not really",
    "I don't think so",
    "Not that I can recall",
    "I haven't noticed that",
    "No, that hasn't happened",
    "I don't experience that",
    "Not particularly",
    "I haven't felt that way",
    "That doesn't sound like me",
    "I haven't had that experience",
]


def generate_utterances(domain: str, is_chr: bool, n_turns: int = 3) -> list:
    """Generate a conversation about a symptom domain.

    Args:
        domain: Symptom domain name
        is_chr: Whether this is a CHR-P case
        n_turns: Number of question/answer turns

    Returns:
        List of utterance dicts
    """
    utterances = []
    questions = PSYCHS_QUESTIONS.get(domain, ["Can you tell me more about that?"])

    for i in range(n_turns):
        # Interviewer question
        question = random.choice(questions)
        utterances.append({
            "speaker": "interviewer",
            "text": question,
            "timestamp": f"00:{random.randint(10, 59)}:{random.randint(10, 59)}",
        })

        # Interviewee response
        if is_chr and random.random() < 0.7:  # 70% chance of CHR response
            response = random.choice(CHR_INDICATORS)
            # Add some detail
            response += f". {generate_detail(domain)}"
        else:
            response = random.choice(CONTROL_RESPONSES)
            if random.random() < 0.3:  # 30% chance of elaboration
                response += f". {generate_control_detail(domain)}"

        utterances.append({
            "speaker": "interviewee",
            "text": response,
            "timestamp": f"00:{random.randint(10, 59)}:{random.randint(10, 59)}",
        })

    return utterances


def generate_detail(domain: str) -> str:
    """Generate a detailed CHR-like response."""
    details = {
        "P1_Unusual_Thoughts": [
            "I sometimes feel like the TV is speaking directly to me.",
            "I notice patterns that others don't seem to see.",
            "I feel like events are connected in meaningful ways.",
        ],
        "P2_Suspiciousness": [
            "I feel like people are watching me when I go out.",
            "I sometimes think others are talking about me behind my back.",
            "I get the sense that I'm being followed sometimes.",
        ],
        "P3_Unusual_Somatic": [
            "I've noticed strange sensations in my body lately.",
            "Sometimes my body feels different, like it's not quite mine.",
            "I've been having odd physical experiences.",
        ],
        "P4_Disoriented": [
            "I lose track of time and forget where I am.",
            "I feel confused about what's happening around me.",
            "Sometimes I can't concentrate on simple tasks.",
        ],
        "P5_Focused_Thoughts": [
            "My thoughts race and jump around quickly.",
            "I can't seem to focus on one thing at a time.",
            "My mind feels scattered and distracted.",
        ],
        "P6_Experiences": [
            "I sometimes hear things that others don't hear.",
            "I've seen things that weren't really there.",
            "I have unusual sensory experiences that concern me.",
        ],
        "P7_Perceiving": [
            "Sounds seem louder or more significant to me.",
            "I notice visual disturbances that others don't see.",
            "My senses seem heightened or altered.",
        ],
        "P8_Confusion": [
            "I feel uncertain about what's real and what's not.",
            "I get confused about everyday situations.",
            "Things that used to make sense now seem puzzling.",
        ],
        "P9_Daydreaming": [
            "I spend a lot of time lost in my own thoughts.",
            "My imagination seems very active lately.",
            "I daydream to escape from reality.",
        ],
        "P10_Time_Experience": [
            "Time seems to move differently for me now.",
            "Hours can feel like minutes or vice versa.",
            "My sense of time feels distorted.",
        ],
        "P11_Depersonalisation": [
            "I sometimes feel disconnected from myself.",
            "It's like I'm watching myself from outside.",
            "I don't always feel like I'm really here.",
        ],
        "P12_Derealisation": [
            "The world sometimes feels like a dream.",
            "Reality seems strange and unfamiliar.",
            "Everything looks different, like it's not quite real.",
        ],
        "P13_Superstitious": [
            "I notice signs and patterns in everyday events.",
            "I feel like certain things are meant to happen.",
            "I have a strong sense of intuition about things.",
        ],
        "P14_Control": [
            "I feel like something is interfering with my thoughts.",
            "It's like my mind isn't entirely my own.",
            "I sense external forces affecting my thinking.",
        ],
        "P15_Reading_Minds": [
            "I sometimes think others can hear my thoughts.",
            "I feel like my mind is being read.",
            "My thoughts seem to be shared with others somehow.",
        ],
    }

    return random.choice(details.get(domain, ["I notice this happening sometimes."]))


def generate_control_detail(domain: str) -> str:
    """Generate a control (non-CHR) detail response."""
    return random.choice([
        "I haven't really noticed anything unusual.",
        "Things have been normal for me.",
        "I don't experience anything like that.",
        "My experiences have been typical.",
        "I haven't had any concerns about that.",
    ])


def generate_transcript(participant_id: str, is_chr: bool) -> dict:
    """Generate a full synthetic transcript.

    Args:
        participant_id: Unique identifier
        is_chr: Whether this is a CHR-P case

    Returns:
        Transcript dict
    """
    # Select a subset of domains (CHR cases have more domains covered)
    all_domains = list(PSYCHS_QUESTIONS.keys())

    if is_chr:
        n_domains = random.randint(8, 15)  # CHR: more domains
    else:
        n_domains = random.randint(3, 7)  # Control: fewer domains

    domains = random.sample(all_domains, n_domains)

    # Generate utterances
    transcript = []
    for domain in domains:
        n_turns = random.randint(2, 4)
        utterances = generate_utterances(domain, is_chr, n_turns)
        transcript.extend(utterances)

    # Add demographics
    age = random.randint(15, 30)
    gender = random.choice(["Male", "Female", "Non-binary"])

    return {
        "participant_id": participant_id,
        "age": age,
        "gender": gender,
        "transcript": transcript,
        "label": "CHR-P" if is_chr else "Healthy",
        "site": f"Site_{random.randint(1, 24)}",
    }


def generate_synthetic_dataset(
    n_participants: int = 100,
    chr_ratio: float = 0.836,  # From paper: 83.6% CHR
    seed: int = 42,
) -> list:
    """Generate synthetic dataset.

    Args:
        n_participants: Number of participants
        chr_ratio: Ratio of CHR-P to Control
        seed: Random seed

    Returns:
        List of transcript dicts
    """
    random.seed(seed)
    np.random.seed(seed)

    n_chr = int(n_participants * chr_ratio)
    n_control = n_participants - n_chr

    data = []

    # Generate CHR-P cases
    for i in range(n_chr):
        participant_id = f"CHR_{i+1:04d}"
        transcript = generate_transcript(participant_id, is_chr=True)
        data.append(transcript)

    # Generate Control cases
    for i in range(n_control):
        participant_id = f"HC_{i+1:04d}"
        transcript = generate_transcript(participant_id, is_chr=False)
        data.append(transcript)

    # Shuffle
    random.shuffle(data)

    return data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic PSYCHS interview data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=100,
        help="Number of participants to generate",
    )
    parser.add_argument(
        "--chr-ratio",
        type=float,
        default=0.836,
        help="Ratio of CHR-P cases (0-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test sets",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.64,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.16,
        help="Validation set ratio",
    )

    args = parser.parse_args()

    # Generate data
    print(f"Generating {args.n_participants} synthetic participants...")
    data = generate_synthetic_dataset(
        n_participants=args.n_participants,
        chr_ratio=args.chr_ratio,
        seed=args.seed,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        # Split data
        from chirpe.data.dataset import split_data

        train_data, val_data, test_data = split_data(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # Save splits
        with open(args.output_dir / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)
        with open(args.output_dir / "val.json", "w") as f:
            json.dump(val_data, f, indent=2)
        with open(args.output_dir / "test.json", "w") as f:
            json.dump(test_data, f, indent=2)

        print(f"Saved train ({len(train_data)}), val ({len(val_data)}), test ({len(test_data)})")
    else:
        # Save all data
        with open(args.output_dir / "all_data.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} transcripts to {args.output_dir / 'all_data.json'}")

    print("Done!")


if __name__ == "__main__":
    main()
