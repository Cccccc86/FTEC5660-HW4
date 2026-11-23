"""Repeat the public experiment with one retrieval change per configuration."""

import argparse
from pathlib import Path
import sys
from functools import partial

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

import hw4


VARIANTS = ('B0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def baseline_chunks(file_name: str, html: str, *, labelled_tables: bool = False) -> list[hw4.Document]:
    """Keep the same recursive splitter when adding labelled tables in V1."""
    soup = BeautifulSoup(html, 'html.parser')
    metadata = hw4.document_metadata(file_name, soup)
    if labelled_tables:
        for table in soup.find_all('table'):
            table.replace_with('\n' + hw4.table_to_text(str(table)) + '\n')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=hw4.MAX_CHUNK_CHARS, chunk_overlap=150,
    )
    return splitter.create_documents([soup.get_text(separator=' ', strip=True)], [metadata])


def configure(variant: str) -> None:
    """Select the ablation condition before calling the unchanged runner."""
    stage = VARIANTS.index(variant)
    if stage < 2:
        hw4.chunk_document = partial(baseline_chunks, labelled_tables=(stage == 1))
    if stage < 3:
        hw4.parse_query = lambda question: {}
    if stage < VARIANTS.index('V6'):
        hw4.build_index = lambda chunks: hw4._make_search_index(chunks, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--data-folder', type=Path, default=Path('public_test'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(f'results_{args.variant}.csv')
    configure(args.variant)
    sys.argv = [sys.argv[0], '--data-folder', str(args.data_folder), '--output', str(output)]
    hw4.main()


if __name__ == '__main__':
    main()
