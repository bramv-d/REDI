import locale
import os
import http.client
import mimetypes
from codecs import encode
from datetime import datetime
import time
import xml.etree.ElementTree as ET

# You should create your own config.py file with the KEY variable
# The key variable is a hash of the username and password for the Tscan.
from config import KEY

source_directory = 'Books'

def loop_through_files(project_name):
    # Walk through the source directory and add the files to the scan
    for root, dirs, files in os.walk(source_directory):
        # Walk through the directories
        for directory in dirs:
            # Walk through the files in the directory
            dir_path = os.path.join(root, directory)
            for sub_root, sub_dirs, sub_files in os.walk(dir_path):
                for file in sub_files:
                    # Get the full path of the file
                    file_path = os.path.join(sub_root, file)
                    # Print the name of the file
                    add_file(project_name, file_path, directory + "_" + file, file)
    return True


def create_project(project_name):
    conn = http.client.HTTPSConnection("tscan.hum.uu.nl")
    payload = ''
    headers = {
        'Authorization': 'Basic ' + KEY
    }
    conn.request("PUT", "/tscan/" + project_name, payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))
    return True


def add_file(project_name, file_path, new_file_name, old_file_name):
    conn = http.client.HTTPSConnection("tscan.hum.uu.nl")
    dataList = []
    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    dataList.append(encode('--' + boundary))
    dataList.append(
        encode('Content-Disposition: form-data; name=file; filename={0}'.format(old_file_name)))

    fileType = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    dataList.append(encode('Content-Type: {}'.format(fileType)))
    dataList.append(encode(''))

    with open(file_path, 'rb') as f:
        dataList.append(f.read())
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=inputtemplate;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("textinput"))
    dataList.append(encode('--' + boundary + '--'))
    dataList.append(encode(''))
    body = b'\r\n'.join(dataList)
    payload = body
    headers = {
        'Authorization': 'Basic ' + KEY,
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
    }
    conn.request("POST", "/tscan/" + project_name + "/input/" + new_file_name, payload, headers)
    res = conn.getresponse()
    data = res.read()
    if res.status != 200:
        print("Error adding file:", new_file_name)
        print(data.decode("utf-8"))
        return
    if res.status == 200:
        print("File added:", new_file_name)


def start_project(project_name):
    conn = http.client.HTTPSConnection("tscan.hum.uu.nl")
    dataList = []
    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=overlapSize;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("50"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=frequencyClip;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("99"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=mtldTreshold;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("0.72"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=useAlpino;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("yes"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=AlpinoOutput;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("no"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=useWopr;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("no"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=sentencePerLine;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("no"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=prevalance;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("The Netherlands"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=word_freq_lex;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("subtlex_words.freq"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=lemma_freq_lex;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("subtlex_lemma.freq"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=top_freq_lex;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("subtlex_words20000.freq"))
    dataList.append(encode('--' + boundary))
    dataList.append(encode('Content-Disposition: form-data; name=compoundSplitterMethod;'))

    dataList.append(encode('Content-Type: {}'.format('text/plain')))
    dataList.append(encode(''))

    dataList.append(encode("never"))
    dataList.append(encode('--' + boundary + '--'))
    dataList.append(encode(''))
    body = b'\r\n'.join(dataList)
    payload = body
    headers = {
        'Authorization': 'Basic ' + KEY,
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
    }
    conn.request("POST", "/tscan/" + project_name + "/", payload, headers)

    res = conn.getresponse()



def poll_project(project_name):
    conn = http.client.HTTPSConnection("tscan.hum.uu.nl")
    boundary = ''
    payload = ''
    headers = {
        'Authorization': 'Basic ' + KEY,
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
    }
    conn.request("GET", "/tscan/" + project_name + "/", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data


def get_results(project_name):
    conn = http.client.HTTPSConnection("tscan.hum.uu.nl")
    boundary = ''
    payload = ''
    headers = {
        'Authorization': 'Basic ' + KEY,
        'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
    }
    conn.request("GET", "/tscan/" + project_name + "/output/total.doc.csv", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data


def save_results_to_file(file_name, project_name):
    # Convert results to utf-8
    data_in_csv = get_results(project_name).decode("utf-8")
    with open(file_name, 'w') as f:
        f.write(data_in_csv)


def poll_for_results(project_name):
    # Poll the project once every 10 seconds until the results are ready
    # Parse the XML
    while True:
        root = ET.fromstring(poll_project(project_name))
        # Find the status element and get its code attribute
        status_code = root.find('.//status').attrib['code']
        if status_code == '2':
            # Print the current time
            # Get the current time
            current_time = datetime.now().time()

            # Print the current time
            print(f"Current time: {current_time}")
            print("Results are ready!")
            return True
        print("Waiting for results...")
        time.sleep(10)



# This methods defines all the seperate stages of getting the t-scan results
def main():
    project_name = "project_name"
    # create_project(project_name)
    # loop_through_files(project_name)
    # start_project(project_name)
    # poll_for_results(project_name)
    save_results_to_file("file_name.csv", project_name)

main()