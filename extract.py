import os
import zipfile
import tarfile


def extract(filename, download_dir):
    """
    Extract the data provided
    Assume the url is a tar file or a zip file

    :param filename: Name of the file with proper file-type-extension
    :param download_dir: Directory where the download file is saved
    """

    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(download_dir):
        print('No directory exist : {}'.format(download_dir))

    print('File found -> extracting..')

    if file_path.endswith('.zip'):
        # unpack the .zip file
        zipfile.ZipFile(file=file_path, mode='r').extractall(download_dir)
    elif file_path.endswith(('.tar.gz', '.tgz')):
        # unpack the tar-file
        tarfile.open(name=file_path, mode='r:gz').extractall(download_dir)

    print('Extraction successfully complete')
