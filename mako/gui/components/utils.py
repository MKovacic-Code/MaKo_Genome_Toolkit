from tkinter import Frame, Label, Entry, StringVar, LEFT, BOTH, X

def add_tab_banner(parent: Frame, title: str, color: str) -> None:
    """Add a colored horizontal banner for tab heading."""
    spacer = Frame(parent, height=6, bg=color)
    spacer.pack(fill="x", padx=10, pady=(4, 6))

def add_labeled_entry(parent: Frame, label: str, variable: StringVar, width: int) -> Entry:
    """Add a labeled entry field with consistent styling."""
    frame = Frame(parent)
    frame.pack(fill=BOTH, pady=2)
    Label(frame, text=label, width=24, anchor="w").pack(side=LEFT)
    entry = Entry(frame, textvariable=variable, width=width)
    entry.pack(side=LEFT, fill=BOTH, expand=True)
    return entry
