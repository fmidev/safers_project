#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage:
# python3 upload_EC.py metadata_HRES_air_temperature_202109130000.json datadir

import sys
import os
import json
import logging

import requests
from requests import HTTPError

from safers_config import OAUTH_URL, CKAN_URL, OAUTH_API_KEY, OAUTH_APP_ID, OAUTH_USER, OAUTH_PWD

REFRESH_TOKEN = ""

# Get access token
def get_access_token():
    url = f'{OAUTH_URL}/api/login'
    body = {
        "loginId": OAUTH_USER,
        "password": OAUTH_PWD,
        "applicationId": OAUTH_APP_ID,
        "noJWT": False
    }
    headers = {
        "Authorization": OAUTH_API_KEY,
    }
    try:
        response = requests.post(url, json=body, headers=headers)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return
    except Exception as err:
        logging.error(f"Other error occurred: {err}")
        return
    else:
        pass
        # logging.info("Access Token obtained")
    if response.status_code == 200:
        REFRESH_TOKEN = response.json()["refreshToken"]
    else:
        logging.error('Error in get_access_token:')
        logging.error(response.json())
        raise Exception
    return response.json()["token"]


def refresh_token():
    url = f'{OAUTH_URL}/jwt/refresh'
    body = {
        "refresh_token": REFRESH_TOKEN
    }
    try:
        response = requests.post(url, data=body)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        logging.error(f"HTTP error occurred in refresh_token: {http_err}")
    except Exception as err:
        logging.error(f"Other error occurred in refresh_token: {err}")
    else:
        pass
        # logging.info("Token refresh")
    return response.json()["token"]


def delete_metadata(metadata_id, access_token=None):
    url = f'{CKAN_URL}/api/action/package_delete'
    if access_token is None:
        access_token = get_access_token()
    headers = {
        "Authorization": f'Bearer {access_token}',
    }
    body = {"id": metadata_id}
    # data = dataset_string.encode('utf-8')
    try:
        response = requests.post(url, json=body, headers=headers)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        logging.error(f"HTTP error occurred in delete_metadata: {http_err}")
        raise Exception
    except Exception as err:
        logging.error(f"Other error occurred in delete_metadata: {err}")
        raise Exception
    else:
        pass
        # logging.info("Access Token obtained")

    if response.status_code == 200:
        response_dict = response.json()
        assert response_dict["success"] is True
    else:
        logging.error('Error in deleting the metadata:')
        logging.error(response.json())
        raise Exception
    return


def upload_metadata(body, access_token=None):
    url = f'{CKAN_URL}/api/action/package_create'
    if access_token is None:
        access_token = get_access_token()
    headers = {
        "Authorization": f'Bearer {access_token}',
    }
    # body['name'] = str(uuid.uuid4())
    # is this needed
    body["spatial"] = str(body["spatial"]).replace("'", '"')
    # data = dataset_string.encode('utf-8')
    try:
        response = requests.post(url, json=body, headers=headers)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        logging.error(f"HTTP error occurred in upload_metadata: {http_err}")
    except Exception as err:
        logging.error(f"Other error occurred in upload_metadata: {err}")
        return
    else:
        pass
        # logging.info("Metadata uploaded")

    if response.status_code == 200:
        response_dict = response.json()
        assert response_dict["success"] is True
        created_package = response_dict["result"]
    else:
        logging.error('Error in upload_metadata:')
        logging.error(response.json())
        raise Exception
    return created_package["id"]


def upload_resource(filepath, body, access_token=None):
    if access_token is None:
        access_token = get_access_token()
    url = f'{CKAN_URL}/api/action/resource_create'
    logging.info(f'Uploading {filepath}')

    headers = {
        "Authorization": f'Bearer {access_token}',
    }
    try:
        with open(filepath, "rb") as file:
            response = requests.post(url,
                                     data=body,
                                     headers=headers,
                                     files=[('upload', file)])
            if response.status_code != 200:
                logging.error(f'Error in upload_resource: {response.status_code}')
                logging.error(response.json())
                raise Exception
    except Exception as e:
        logging.error("Error occurred in upload_resource: " + str(e))
        raise
    return


def upload_nc(metadatafile, ncfile, access_token=None):
    """Upload one netcdf file to SAFERS datalake."""
    with open(metadatafile, "r", encoding='UTF-8') as f:
        metadata_body = json.load(f)

    # read file names and other info from metadata attributes
    #ncfile = metadata_body['external_attributes']['file']
    fctimes = metadata_body['external_attributes']['fctimes']
    #times2 = metadata_body['external_attributes']['fctimes_end']
    origintime = metadata_body['external_attributes']['origintime']
    dataid = metadata_body['external_attributes']['datatype_resource']
    dataformat = metadata_body['external_attributes']['format']

    name = metadata_body['external_attributes'].get('name')
    if name is None:
        name = os.path.basename(ncfile)
    # get rid of extra metadata before upload
    #del metadata_body['external_attributes']['fcfiles']
    del metadata_body['external_attributes']['fctimes']
    #del metadata_body['external_attributes']['fctimes_end']
    del metadata_body['external_attributes']['datatype_resource']

    if access_token is None:
        access_token = get_access_token()

    # upload metadata
    metadata_id = upload_metadata(metadata_body, access_token)
    logging.info(f'metadata_id: {metadata_id}')

    # upload datasets
    time1 = fctimes[0]
    time2 = fctimes[-1]
    body = {
        "package_id": metadata_id,
        "name": name,
        "format": dataformat,
        "datatype_resource": dataid,
        "origintime": origintime,
        "file_date_start": time1,
        "file_date_end": time2,
    }
    # print(body)
    try:
        upload_resource(ncfile, body, access_token)
    except:
        logging.error('NC dataset upload failed. Removing the metadata...')
        # write to stderr to get mail from crond
        print('NC dataset upload failed. Removing the metadata...', file=sys.stderr)
        delete_metadata(metadata_id, access_token)
        print(f'Upload failed for {ncfile}', file=sys.stderr)

    logging.info("DONE")


def upload_files(metadata_filename, resources_filepath, access_token=None):
    """Upload geoJSON files to SAFRES datalake."""
    # Read JSON data into the metadata_body variable
    with open(metadata_filename, "r", encoding='UTF-8') as f:
        metadata_body = json.load(f)

    # read file names and other info from metadata attributes
    files = metadata_body['external_attributes']['fcfiles']
    times = metadata_body['external_attributes']['fctimes']
    times2 = metadata_body['external_attributes']['fctimes_end']
    origintime = metadata_body['external_attributes']['origintime']
    dataids = metadata_body['external_attributes']['datatype_resource']
    dataformat = metadata_body['external_attributes']['format']
    nfiles = len(files)

    # get rid of extra metadata before upload
    del metadata_body['external_attributes']['fcfiles']
    del metadata_body['external_attributes']['fctimes']
    del metadata_body['external_attributes']['fctimes_end']
    del metadata_body['external_attributes']['datatype_resource']

    if access_token is None:
        access_token = get_access_token()

    # upload metadata
    metadata_id = upload_metadata(metadata_body, access_token)
    logging.info(f'metadata_id: {metadata_id}')

    # upload datasets
    try:
        for i in range(nfiles):
            if isinstance(dataids, list):
                dataid = dataids[i]
            else:
                dataid = dataids
            file = files[i]
            time1 = times[i]
            time2 = times2[i]
            body = {
                "package_id": metadata_id,
                "name": os.path.basename(file),
                "format": dataformat,
                "datatype_resource": dataid,
                "origintime": origintime,
                "file_date_start": time1,
                "file_date_end": time2,
            }
            # print(body)
            try:
                upload_resource(os.path.join(resources_filepath, file), body, access_token)
            except Exception:
                logging.error('Some error. Removing the metadata...')
                delete_metadata(metadata_id, access_token)
                return

        logging.info("DONE")
    except Exception:
        # if any error occurs in uploading datasets, delete the metadata
        # (otherwise it will be metadata with no data)
        logging.error('One or more dataset upload failed. Removing the metadata...')
        delete_metadata(metadata_id, access_token)


# main routine
if __name__ == "__main__":

    logging.basicConfig(level='DEBUG')

    # set metadata path and resources path
    # metadata_filename = ""
    # resources_filepath = "/data/tmp/safers_geojson"
    args = sys.argv
    if len(args) > 2:
        metadata_filename = args[1]
        nc_filename = args[2]
    else:
        logging.error('ERROR, give metadata file and nc file as argument')
        sys.exit(1)
    #if len(args) > 2:
    #    resources_filepath = args[2]

    access_token = get_access_token()
    upload_nc(metadata_filename, nc_filename, access_token)
