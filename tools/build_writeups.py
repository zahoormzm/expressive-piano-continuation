"""Build the architecture-focused one-page and two-page project write-ups."""

from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
TITLE = "Piano music continuation with Transformers"
AUTHORS = (
    "Mohammed Zahoor Mashahir - PES1UG24CS579<br/>"
    "Mohammed Ibrahim - PES1UG24CS578<br/>"
    "UE24CS352A Machine Learning | 13 October 2026"
)
INTRO = (
    "Our project generates a continuation of a solo-piano MIDI prompt. We use two versions of "
    "our Transformer: an initially trained model and a final model obtained after post-training. "
    "They share the same network architecture. The distinction is how their weights have been "
    "trained. A user supplies a musical starting idea, and the selected checkpoint predicts "
    "the events that follow it. The system exports the result as editable MIDI and piano audio."
)
WHY = (
    "We chose a decoder-only Transformer because music is an ordered sequence with relationships "
    "across many events. A later note can relate to an earlier chord, rhythm or melodic phrase. "
    "Self-attention lets a token use information from earlier positions within the available "
    "context. A causal mask permits only the current and previous positions, matching the task "
    "of predicting the next token without seeing the future. This makes the same network useful "
    "for learning from complete performances and continuing a supplied prompt. Music Transformer "
    "research motivates this use of self-attention [1]."
)
REPRESENTATION = (
    "We use MAESTRO piano MIDI performances and keep training, validation and test splits separate. "
    "MIDI describes notes and performance timing directly, so the model learns musical events "
    "rather than audio waveforms. Each piano note uses a pitch-and-velocity token followed by "
    "onset and duration tokens. Timing uses 10 ms steps and five-second segments. Prefix and "
    "boundary tokens identify the instrument and sequence structure. This representation exposes "
    "melody, rhythm and dynamics while keeping the generated music editable."
)
ARCHITECTURE_ROWS = [
    ["Component", "Configuration", "Purpose"],
    [
        "Input and context",
        "17,727 token IDs;\n2,048-token context",
        "Represent events and bound attention cost.",
    ],
    ["Embedding", "512 dimensions", "Turn each token ID into a learned feature vector."],
    [
        "Decoder stack",
        "8 blocks; 8 heads\nper attention layer",
        "Combine relationships among earlier events.",
    ],
    ["Position encoding", "Rotary positions (RoPE)", "Encode token order in attention [2]."],
    ["Feed-forward layer", "SwiGLU; width 1,536", "Transform features through a learned gate [4]."],
    [
        "Stability and regularization",
        "Pre-LayerNorm, residual\nconnections, dropout 0.1",
        "Support gradient flow and regularize training [3].",
    ],
    [
        "Output head",
        "Final LayerNorm;\ntied embedding weights",
        "Produce next-token scores while reusing parameters.",
    ],
]
INSIDE_BLOCK = (
    "Inside each block, normalization precedes causal multi-head attention. The block adds the "
    "attention output back to its input through a residual connection. A second normalization "
    "feeds SwiGLU, whose SiLU-activated gate controls a parallel feature projection; another "
    "residual connection adds that output. RoPE acts on queries and keys. Token order and musical "
    "time are distinct: explicit onset and duration tokens carry time. The 36.36M-parameter "
    "design balances expressive capacity with our local 12 GB GPU budget."
)
INITIAL = (
    "The initial Transformer starts with randomly initialized weights and learns next-token "
    "prediction from piano performances. Token sequences are cropped within a single piece "
    "and end on complete note groups. Inputs and targets are shifted by one position: each "
    "position predicts the token that follows it. Cross-entropy trains the predicted distribution "
    "against the real next token. Padding and the supplied instrument/start prefix do not "
    "contribute to loss. This stage builds the model's general knowledge of the event "
    "representation and the musical patterns present in the corpus."
)
POST = (
    "The final Transformer descends from the initial checkpoint through further training. "
    "We retain the decoder stack, vocabulary, context length and output head, then continue "
    "updating the learned weights. Post-training includes examples tailored to the continuation "
    "task. In continuation-focused examples, the model receives a short musical prompt as context "
    "and the loss scores the tokens after that prompt. We choose complete onset groups so a "
    "chord stays intact and leave enough following notes as targets. Ordinary next-token "
    "examples remain in the training mix, preserving the general prediction task."
)
AUGMENTATION = (
    "We also augment whole passages with one consistent pitch shift and one consistent time "
    "scale. Moving all notes by the same number of semitones preserves intervals and chord "
    "relationships. Scaling onsets and durations together preserves their timing relationships "
    "up to token quantization. Shifts must keep every pitch in the piano range. These examples "
    "give the model different keys and timing scales for related musical structures. The purpose "
    "is to align training with continuing an existing idea; the architecture stays unchanged."
)
OPTIMIZATION = (
    "Both stages use AdamW and gradient clipping. BF16 and gradient accumulation let us train "
    "with the available GPU memory. The final stage uses a lower learning rate so updates are "
    "smaller around the already learned solution. Checkpoints preserve the weights, optimizer "
    "and random states so interrupted training can resume consistently."
)
GENERATION = (
    "At inference, the system tokenizes the user's prompt and passes it to either checkpoint. "
    "The output head gives scores for the next token. Temperature and min-p sampling control "
    "variation, while event constraints keep note groups valid. The chosen token joins the "
    "sequence and becomes context for the next prediction. This repeats until an end token, "
    "the requested onset horizon or the context limit stops generation. We decode the events "
    "to MIDI and use a fixed piano renderer for audio. Note releases can continue after the "
    "last generated onset. The Transformer models event structure and expression; the renderer "
    "provides the piano sound."
)
CONTRIBUTIONS = (
    "Equal contribution allocation: Mohammed Zahoor Mashahir covers data preparation, "
    "tokenization and Transformer training. Mohammed Ibrahim covers generation, piano rendering, "
    "the application and submission documentation. Both share method interpretation and "
    "demonstration preparation. Development used an AI coding assistant."
)
REFERENCES = [
    "[1] Huang et al. Music Transformer. https://arxiv.org/abs/1809.04281",
    "[2] Su et al. RoFormer. https://arxiv.org/abs/2104.09864",
    "[3] Xiong et al. On Layer Normalization in the Transformer Architecture. https://arxiv.org/abs/2002.04745",
    "[4] Shazeer. GLU Variants Improve Transformer. https://arxiv.org/abs/2002.05202",
    (
        "Data: MAESTRO, https://magenta.tensorflow.org/datasets/maestro (CC BY-NC-SA 4.0). "
        "Implementation: src/music_project/models/networks.py; configs/transformer_full.yaml; "
        "configs/transformer_musical.yaml."
    ),
]


def styles(compact=False):
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=9
        ),
        "meta": ParagraphStyle(
            "meta", fontName="Helvetica", fontSize=9.3, leading=12.5, spaceAfter=9
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=9,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9.8 if compact else 10.2,
            leading=13.1 if compact else 14,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=8.2, leading=10.8, spaceAfter=3
        ),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=11.8),
        "header": ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=9, leading=11.8),
    }


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5a675f"))
    canvas.drawString(
        43, 25, "UE24CS352A Machine Learning    Transformer architecture and training"
    )
    canvas.drawRightString(A4[0] - 43, 25, str(doc.page))


def build_full():
    st = styles()

    def p(value, kind="body"):
        return Paragraph(value, st[kind])

    rows = [
        [p(value.replace("\n", "<br/>"), "header" if i == 0 else "cell") for value in row]
        for i, row in enumerate(ARCHITECTURE_ROWS)
    ]
    table = Table(rows, colWidths=[119, 146, 244], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e9e3")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d9d9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story = [
        p(TITLE, "title"),
        p(AUTHORS, "meta"),
        p(INTRO),
        p("Why we use a Transformer", "heading"),
        p(WHY),
        p("Musical input representation", "heading"),
        p(REPRESENTATION),
        p("Our decoder architecture", "heading"),
        table,
        Spacer(1, 7),
        p(INSIDE_BLOCK),
        PageBreak(),
        p("The initial Transformer", "heading"),
        p(INITIAL),
        p("The final Transformer after post-training", "heading"),
        p(POST),
        p(AUGMENTATION),
        p("Optimization and checkpointing", "heading"),
        p(OPTIMIZATION),
        p("How the architecture generates music", "heading"),
        p(GENERATION),
        p("Contributions", "heading"),
        p(CONTRIBUTIONS),
        p("References", "heading"),
    ]
    story += [p(reference, "small") for reference in REFERENCES]
    destination = OUT / "project-writeup.pdf"
    SimpleDocTemplate(
        str(destination), pagesize=A4, leftMargin=43, rightMargin=43, topMargin=35, bottomMargin=38
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    assert len(PdfReader(destination).pages) == 2


def build_compact():
    st = styles(compact=True)

    def p(value, kind="body"):
        return Paragraph(value, st[kind])

    why = (
        "Self-attention lets a musical event use earlier notes, chords and rhythmic patterns "
        "within the context. Causal masking hides future tokens, so training and generation "
        "both follow next-token prediction. This matches a continuation task in which the "
        "supplied prompt is the beginning of the sequence [1]."
    )
    architecture = (
        "Piano MIDI becomes pitch/velocity, onset and duration tokens on a 10 ms timing grid. "
        "A 17,727-entry vocabulary feeds 512-dimensional embeddings and eight decoder blocks "
        "with eight attention heads each. RoPE encodes token order in queries and keys [2]. "
        "Each block uses Pre-LayerNorm, residual connections, dropout 0.1 and a SwiGLU "
        "feed-forward layer of width 1,536 [3,4]. Final normalization and a tied output head "
        "produce next-token scores. Shared embedding/output weights save parameters. The "
        "2,048-token context and 36.36M parameters balance modeling capacity with a 12 GB GPU."
    )
    initial = (
        "The initial model learns from MAESTRO piano performances with separate data splits. "
        "Its weights start randomly. Inputs and targets shift by one token, and cross-entropy "
        "teaches the model to predict the next event token. Piece-specific crops preserve "
        "complete note groups. Padding and supplied instrument/start tokens do not contribute "
        "to loss. AdamW, BF16 and gradient accumulation support training on our hardware."
    )
    final = (
        "The final model continues from the initially trained weights while keeping the same "
        "architecture. Post-training combines whole-passage pitch/time augmentation with "
        "continuation-focused examples. A shared pitch shift preserves intervals and chords; "
        "a shared time scale transforms onsets and durations together. Prompt tokens provide "
        "context while the loss scores the following music. Ordinary prediction examples "
        "remain in the mix. Smaller learning-rate updates adapt the learned weights to the "
        "prompt-continuation task."
    )
    generation = (
        "Either checkpoint receives the tokenized prompt and generates one token at a time. "
        "Sampling controls variation and event constraints keep note groups valid. Generated "
        "tokens return as context until EOS, the onset horizon or the context limit. Decoded "
        "MIDI supplies the notes; a fixed renderer produces the piano audio."
    )
    contribution = (
        "Equal allocation: Zahoor covers data, tokenization and Transformer training; Ibrahim "
        "covers generation, rendering, the application and documentation. Both share method "
        "interpretation and demo preparation. Development used an AI coding assistant."
    )
    story = [
        p(TITLE, "title"),
        p(AUTHORS, "meta"),
        p(INTRO),
        p("Why a Transformer", "heading"),
        p(why),
        p("Architecture", "heading"),
        p(architecture),
        p("Initial model", "heading"),
        p(initial),
        p("Final model after post-training", "heading"),
        p(final),
        p("Generation and contributions", "heading"),
        p(generation),
        p(contribution, "small"),
        p("References", "heading"),
        p(
            "[1] Huang et al., Music Transformer (1809.04281). [2] Su et al., RoFormer "
            "(2104.09864). [3] Xiong et al., Layer Normalization (2002.04745). [4] Shazeer, "
            "GLU Variants (2002.05202). Papers: arxiv.org. MAESTRO: "
            "magenta.tensorflow.org/datasets/maestro, CC BY-NC-SA 4.0.",
            "small",
        ),
    ]
    destination = OUT / "project-writeup-one-page.pdf"
    SimpleDocTemplate(
        str(destination), pagesize=A4, leftMargin=43, rightMargin=43, topMargin=32, bottomMargin=38
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    assert len(PdfReader(destination).pages) == 1


def write_markdown():
    sections = [
        ("Project overview", INTRO),
        ("Why we use a Transformer", WHY),
        ("Musical input representation", REPRESENTATION),
        ("Inside the decoder", INSIDE_BLOCK),
        ("The initial Transformer", INITIAL),
        ("The final Transformer after post-training", POST + "\n\n" + AUGMENTATION),
        ("Optimization and checkpointing", OPTIMIZATION),
        ("How the architecture generates music", GENERATION),
        ("Contributions", CONTRIBUTIONS),
    ]
    text = "# " + TITLE + "\n\n" + AUTHORS.replace("<br/>", "\n\n") + "\n\n"
    for title, content in sections:
        text += "## " + title + "\n\n" + content + "\n\n"
        if title == "Musical input representation":
            text += "## Architecture configuration\n\n"
            text += "| Component | Configuration | Purpose |\n| --- | --- | --- |\n"
            for row in ARCHITECTURE_ROWS[1:]:
                text += "| " + " | ".join(value.replace("\n", " ") for value in row) + " |\n"
            text += "\n"
    text += "## References\n\n" + "\n\n".join(REFERENCES) + "\n"
    (OUT / "project-writeup.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_full()
    build_compact()
    write_markdown()
    print("Built two-page and one-page write-ups focused on Transformer design and training.")
