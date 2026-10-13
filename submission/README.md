# Project submission

Expressive piano continuation for UE24CS352A Machine Learning.

Students: Mohammed Zahoor Mashahir (PES1UG24CS579) and Mohammed Ibrahim (PES1UG24CS578). The user requested equal contribution allocation. The confirmed submission date is 13 October 2026.

## Required files

- `project-writeup.pdf`: two-page explanation of the Transformer choice, architecture, initial training and final post-training stage, matching the body of the assignment summary.
- `project-writeup-one-page.pdf`: compact alternative because the supplied plan reports a one-page heading and two-page body. Confirm which version faculty accepts.
- `project-writeup.md`: readable source of the architecture-focused explanation. `tools/build_writeups.py` rebuilds both PDFs with ReportLab and pypdf.
- `project-slides.pptx`: editable presentation covering the task, implementation, results, live demo, limits and contribution allocation.
- `contributions.md`: equal allocation of responsibilities and the shared assessment work.
- `artifacts-manifest.json`: hashes and download locations for the trained models, preparation artifacts and listening evidence in the private release.
- `../README.md`: exact setup, training, recovery and local app commands.
- `../docs/demo-guide.md`: live demonstration and rehearsal procedure.
- `reference/Music_Generation_Project_Implementation_Plan.docx`: supplied planning reference, preserved unchanged.

## Access and demonstration

The repository is private. The owner must grant access to the assigned faculty/TAs using their GitHub usernames. No reviewer identities were supplied in this conversation, and no invitations have been sent.

Follow the artifact restoration instructions in `../docs/artifact-handoff.md`. Launch the local studio with `python scripts/serve_app.py` in the configured project environment. A local application and saved listening suite are available. A human rehearsal and actual submission are separate actions, not implied by this repository upload.

## Results and remaining scope

The original LSTM/Transformer comparison uses matching 100M-target budgets. Later Transformer runs use additional compute. The final musical candidate uses continuation masks, so its cumulative target count is bookkeeping across different objectives. The unmodified upstream Aria baseline has different training exposure. No project Aria LoRA adaptation result exists. The small informal listening feedback does not establish broad musical superiority, and the final musical candidate still needs human listening.
