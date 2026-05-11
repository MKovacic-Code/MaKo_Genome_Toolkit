# MaKo Genome Toolkit

MaKo Genome Toolkit is a powerful, high-performance suite for searching and analyzing genomic sequences. It supports complex nucleotide motif searches, translated peptide searches, and combined search strategies that evaluate constraints across both levels simultaneously.

## Features

- **Nucleotide Search**: Advanced scanning with IUPAC support, base content filtering, and repeat analysis.
- **Peptide Search**: Scan translated ORFs with support for amino acid content percentage and consecutive repeat filtering.
- **Combined Search**: Synchronized scanning that filters regions satisfying both DNA and Protein constraints.
- **Bi-directional Sync**: Real-time synchronization of search parameters across different analytical tabs.
- **Visualization**: Integrated chromosome and dataset visualization tools.

## Installation

### Prerequisites

- **Python 3.10+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MaKo-Genome-Toolkit.git
   cd MaKo-Genome-Toolkit
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Compile C-extensions (Optional but recommended for performance)**:
   ```bash
   python setup.py build_ext --inplace
   ```

## Usage

Run the main GUI application:
```bash
python nucleotide_gui.pyw
```

## Genome Data

The toolkit is designed to work with genomic FASTA files.

### Downloading Data
You can download genome assemblies from sources like:
- [NCBI Genome Database](https://www.ncbi.nlm.nih.gov/genome/)
- [Ensembl](https://www.ensembl.org/info/data/ftp/index.html)
- [UCSC Genome Browser](http://hgdownload.soe.ucsc.edu/downloads.html)

### Organizing Data
For the toolkit to automatically discover genomes, place them in directories named with the prefix `data_` in the project root. For example:
```
MaKo-Genome-Toolkit/
├── data_human_hg38/
│   └── GRCh38.fna
├── nucleotide_gui.pyw
└── ...
```

## Requirements

The core dependencies are:
- `biopython`: For biological sequence processing.
- `matplotlib`: For visualization.
- `requests`: For downloading data (if using remote scripts).
- `cython`: For high-performance sequence scanning extensions.

## License

Open Source
