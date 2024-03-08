#!/usr/bin/env python

import argparse
import bz2
import csv
import hashlib
import os
import shlex
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str,
                        help='File containing cache files to compile '
                             'in the format of: filename, args')
    parser.add_argument('--num-cus', default=4, type=int,
                        help='Number of CUs in simulated GPU')
    parser.add_argument('--gfx-version', default='gfx902',
                        choices=['gfx900', 'gfx902'],
                        help='gfx version of simulated GPU')

    return parser.parse_args()


def getDb(options):
    db_name = f'{options.gfx_version}_{options.num_cus}.ukdb'
    db_path = '/root/.cache/miopen/2.9.0/'

    full_db_path = os.path.join(db_path, db_name)
    # Should create file if it doesn't exist
    # Does assume db_path exists, which it should in the Docker image
    con = sqlite3.connect(full_db_path)

    cur = con.cursor()

    # Ripped from src/include/miopen/kern_db.hpp
    cur.execute('''CREATE TABLE IF NOT EXISTS kern_db (
                        id INTEGER PRIMARY KEY ASC,
                        kernel_name TEXT NOT NULL,
                        kernel_args TEXT NOT NULL,
                        kernel_blob BLOB NOT NULL,
                        kernel_hash TEXT NOT NULL,
                        uncompressed_size INT NOT NULL);''')
    cur.execute('''CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_kern_db ON kern_db (kernel_name, kernel_args);''')

    return con


def insertFiles(con, options):
    miopen_kern_path = '/MIOpen/src/kernels'

    extra_args = {'gfx900': '-Wno-everything -Xclang '
                            '-target-feature -Xclang +code-object-v3',
                  'gfx902': '-Wno-everything -Xclang '
                            '-target-feature -Xclang +code-object-v3'}

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(options.csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                miopen_kern = row[0]
                miopen_kern_full = os.path.join(miopen_kern_path, miopen_kern)
                # We want to manually add the gfx version
                # Additionally, everything after the gfx version isn't
                # used in the database
                # Explicitly add the leading space because that's used
                # in the database
                args = (f' {row[1].split("-mcpu")[0].strip()} '
                        f'-mcpu={options.gfx_version}')

                # Hash to generate unique output files
                file_hash = hashlib.md5(args.encode('utf-8')).hexdigest()
                outfile = f'{miopen_kern}-{file_hash}.o'
                full_outfile = os.path.join(tmpdir, outfile)

                # Compile the kernel
                cmd_str = (f'/opt/rocm/bin/clang-ocl {args} '
                           f'{extra_args[options.gfx_version]} '
                           f'{miopen_kern_full} -o {full_outfile}')
                cmd_args = shlex.split(cmd_str)
                subprocess.run(cmd_args, check=True)

                # Get other params needed for db
                uncompressed_file = open(full_outfile, 'rb').read()
                uncompressed_size = Path(full_outfile).stat().st_size
                uncompressed_hash = hashlib.md5(uncompressed_file).hexdigest()
                compressed_blob = bz2.compress(uncompressed_file)

                cur = con.cursor()
                cur.execute('''INSERT OR IGNORE INTO kern_db
                               (kernel_name, kernel_args, kernel_blob, kernel_hash, uncompressed_size)
                               VALUES(?, ?, ?, ?, ?)''',
                            (f'{miopen_kern}.o', args, compressed_blob,
                                uncompressed_hash, uncompressed_size))


def main():

    args = parseArgs()

    con = getDb(args)

    insertFiles(con, args)

    con.commit()
    con.close()


if __name__ == '__main__':
    main()
