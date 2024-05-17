"""S3 utilities for SAFERS."""

import os
import re
from datetime import datetime

import sqlite3
import logging

import numpy as np
import pandas as pd

import boto3
# import botocore
# from botocore.exceptions import ClientError

from safers_config import S3_KEY, S3_SECRET

DEFAULT_BUCKET = 'safers-ecmwf'

S3_DATABASE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'safers_ecmwf_uploads.sqlite')


class SQLtimestamp():
    """Store timestaps in a database."""

    init_sql = """CREATE TABLE IF NOT EXISTS latest(
                         fctype text primary key,
                         origintime timestamp,
                         processtime timestamp
                  );"""

    put_sql = """INSERT OR REPLACE INTO latest(
                        fctype,
                        origintime,
                        processtime
                 )
                 VALUES (
                      '%(fctype)s',
                      '%(origintime)s',
                      '%(processtime)s'
                 );"""

    get_sql = """SELECT fctype, origintime FROM latest
                  WHERE fctype == '%(fctype)s'"""

    getall_sql = """SELECT * FROM latest"""

    def __init__(self, fctype='HRES', dbase=S3_DATABASE):
        self.fctype = fctype
        self.dbase = dbase
        conn = sqlite3.connect(self.dbase,
                               detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.execute(SQLtimestamp.init_sql)
        conn.commit()
        self.conn = conn

    def put(self, time):
        """Inset timestamp into database."""
        cur = self.conn.cursor()
        cur.execute(SQLtimestamp.put_sql %
                    {'fctype': self.fctype,
                     'origintime': pd.to_datetime(time),
                     'processtime': pd.to_datetime(time)})
        self.conn.commit()
        cur.close()

    def get(self):
        """Get timestamp."""
        cur = self.conn.cursor()
        cur.execute(SQLtimestamp.get_sql % {'fctype': self.fctype})
        r = cur.fetchone()
        cur.close()
        if r is None:
            return r
        else:
            return r[1]

    def df(self):
        """Database to dataframe."""
        query = self.conn.execute(SQLtimestamp.getall_sql)
        cols = [column[0] for column in query.description]
        df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        return df


def s3_client(bucket=DEFAULT_BUCKET):
    """Create S3 client to access the files."""
    s3 = boto3.client("s3",
                      aws_access_key_id=S3_KEY,
                      aws_secret_access_key=S3_SECRET,
                      endpoint_url='https://lake.fmi.fi')
    return s3


def s3_get_filenames(d='hres/', s3=None, bucket=DEFAULT_BUCKET):
    """Return a pandas data frame containing files in the bucket."""
    if s3 is None:
        s3 = s3_client(bucket)
    objects = s3.list_objects(Bucket=bucket,
                              Prefix=d,
                              Delimiter='/')
    # contents = objects['Contents']
    contents = objects.get('Contents')
    if contents is None:
        return None
    files = [contents[i]['Key'] for i in range(len(contents))]
    # only fc*.grib
    files = list(filter(lambda x: re.search(f'^{d}fc.*grib$', x), files))
    df = s3_contents_HRES(files)
    return df


def latest_time(d='hres/', s3=None, bucket=DEFAULT_BUCKET, df=None):
    """Latest forecast time in s3."""
    if df is None:
        df = s3_get_filenames(d=d, s3=s3, bucket=bucket)
    if df is None:
        return np.datetime64('1970-01-01')
    else:
        d = df['origintime'].values
        d.sort()
    return d[-1]


def s3_copy_file(src, target, s3=None, bucket=DEFAULT_BUCKET):
    """Copy file from S3 to target file."""
    if os.path.exists(target):
        logging.info('Target already exists, will not overwrite: %s', target)
        return
    if s3 is None:
        s3 = s3_client(bucket)
    s3.download_file(bucket, src, target)


# fcYYYYMMDDhhmm
def parsefile_HRES(f):
    """Parse forecast filename to get fc times."""
    f1 = os.path.basename(f)
    f0 = os.path.splitext(f1)[0]
    time = datetime.strptime(f0[-12:], '%Y%m%d%H%M')
    return([f, f1, time])

def parsefile_ENS(f):
    """Parse forecast filename to get fc times."""
    f1 = os.path.basename(f)
    f0 = os.path.splitext(f1)[0]
    time = datetime.strptime(f0[-12:], '%Y%m%d%H%M')
    return([f, f1, time])

def parsefile_ENS_calibrix(f):
    """Parse enseble forecast file name."""
    f1 = os.path.basename(f)
    f0 = os.path.splitext(f1)[0]
    mbr = f0[-3:]
    time = datetime.strptime(f0[-19:-7], '%Y%m%d%H%M')
    return([f, f1, time, mbr])


def s3_contents_HRES(files):
    """Generate data frame from files."""
    df = pd.DataFrame([parsefile_HRES(f) for f in files],
                      columns=['key', 'file', 'origintime'])
    return df


def s3_contents_ENS(files):
    """Generate data frame from files."""
    df = pd.DataFrame([parsefile_ENS(f) for f in files],
                      columns=['key', 'file', 'origintime'])
#                      columns=['key', 'file', 'time', 'mbr'])
    return df
