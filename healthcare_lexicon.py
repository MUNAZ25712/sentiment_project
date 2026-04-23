"""
healthcare_lexicon.py
Enhanced healthcare-specific terms for better accuracy
"""

# Medical outcomes - Positive
POSITIVE_OUTCOMES = [
    'recovered', 'healed', 'improved', 'better', 'relieved', 'cured',
    'successful', 'effective', 'worked', 'helped', 'beneficial',
    'saved my life', 'life-saving', 'walk again', 'feel better',
    'back to normal', 'fully recovered', 'quick recovery'
]

# Medical outcomes - Negative
NEGATIVE_OUTCOMES = [
    'worsened', 'complicated', 'failed', 'ineffective', 'unsuccessful',
    'misdiagnosed', 'wrong diagnosis', 'sent me home', 'made me worse',
    'caused more pain', 'did not help', 'no improvement', 'got sicker'
]

# Doctor qualities - Positive
DOCTOR_POSITIVE = [
    'compassionate', 'caring', 'attentive', 'thorough', 'knowledgeable',
    'experienced', 'skilled', 'professional', 'expert', 'brilliant',
    'explained', 'listened', 'answered', 'clarified', 'understood'
]

# Doctor qualities - Negative
DOCTOR_NEGATIVE = [
    'dismissive', 'arrogant', 'condescending', 'negligent', 'incompetent',
    'inexperienced', 'unskilled', 'unqualified', 'careless',
    'ignored', 'interrupted', 'rushed', 'brushed off'
]

# Staff qualities - Positive
STAFF_POSITIVE = [
    'friendly', 'helpful', 'courteous', 'respectful', 'efficient',
    'prompt', 'responsive', 'accommodating', 'understanding'
]

# Staff qualities - Negative
STAFF_NEGATIVE = [
    'rude', 'unhelpful', 'unfriendly', 'disrespectful', 'slow',
    'unresponsive', 'inattentive', 'neglectful', 'abandoned'
]

# Facility qualities - Positive
FACILITY_POSITIVE = [
    'clean', 'modern', 'well-maintained', 'comfortable', 'spacious',
    'organized', 'efficient', 'quiet', 'peaceful'
]

# Facility qualities - Negative
FACILITY_NEGATIVE = [
    'dirty', 'outdated', 'poorly maintained', 'uncomfortable', 'cramped',
    'disorganized', 'chaotic', 'noisy', 'crowded'
]

# Wait time phrases
WAIT_POSITIVE = [
    'short wait', 'no wait', 'quick service', 'on time', 'prompt',
    'efficient', 'fast service'
]

WAIT_NEGATIVE = [
    'long wait', 'excessive wait', 'hours waiting', 'delayed',
    'late appointment', 'waited forever', 'took too long'
]

# Negation patterns (for complex cases)
NEGATION_PATTERNS = {
    'neutral_mixed': [
        r'not\s+\w+,\s+but\s+not\s+\w+',  # "not X, but not Y"
        r'not\s+(terrible|bad|awful).*not\s+(great|good|excellent)'
    ],
    'negative': [
        r'didn\'?\s+work\s+as\s+expected',
        r'didn\'?\s+help\s+at\s+all',
        r'waste\s+of\s+(time|money)'
    ],
    'positive': [
        r'no\s+issues?\s+with',
        r'no\s+complaints?\s+about',
        r'couldn\'?\s+ask\s+for\s+better'
    ]
}

# Combine all positive terms
ALL_POSITIVE = set(POSITIVE_OUTCOMES + DOCTOR_POSITIVE + STAFF_POSITIVE + 
                   FACILITY_POSITIVE + WAIT_POSITIVE)

# Combine all negative terms
ALL_NEGATIVE = set(NEGATIVE_OUTCOMES + DOCTOR_NEGATIVE + STAFF_NEGATIVE + 
                   FACILITY_NEGATIVE + WAIT_NEGATIVE)

# Neutral phrases (high priority)
NEUTRAL_PHRASES = [
    'nothing special', 'not special', 'nothing great', 'nothing bad',
    'nothing good', 'nothing terrible', 'neither good nor bad',
    'not good not bad', 'so-so', 'just okay', 'just fine', 'just average',
    'average experience', 'nothing to complain about', 'nothing to praise',
    'could be better could be worse', 'no strong feelings', 'mixed feelings',
    'decent but not great', 'acceptable but not exceptional'
]