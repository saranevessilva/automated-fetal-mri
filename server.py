
import constants
from connection import Connection

import socket
import logging
import multiprocessing
import ismrmrd.xsd
import importlib
import os
import simplefft
import invertcontrast
import autoplan_fetal_cardiac_landmarks
import automated_volumetry
import autoplan_fetal_sagittal_landmarks
import autoplan_fetal_head_landmarks
import autoplan_fetal_head_landmarks_offline
import autoplan_fetal_head_landmarks_phantom
import autoplan_fetal_head_landmarks_phantom_offline
import automated_svr_launch
import automated_pull_svr_brain
import automated_pull_svr_body
import mrd2dicom


class Server:
    """
    Something docstring.
    """

    def __init__(self, address, port, savedata, savedataFolder, multiprocessing):
        logging.info("Starting server and listening for data at %s:%d", address, port)
        if (savedata is True):
            logging.debug("Saving incoming data is enabled.")

        if (multiprocessing is True):
            logging.debug("Multiprocessing is enabled.")

        self.multiprocessing = multiprocessing
        self.savedata = savedata
        self.savedataFolder = savedataFolder
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))

    def serve(self):
        logging.debug("Serving... ")
        self.socket.listen(0)

        while True:
            sock, (remote_addr, remote_port) = self.socket.accept()

            logging.info("Accepting connection from: %s:%d", remote_addr, remote_port)

            if (self.multiprocessing is True):
                process = multiprocessing.Process(target=self.handle, args=[sock])
                process.daemon = True
                process.start()
                logging.debug("Spawned process %d to handle connection.", process.pid)
            else:
                self.handle(sock)

    def handle(self, sock):

        try:
            connection = Connection(sock, self.savedata, "", self.savedataFolder, "dataset")

            # First message is the config (file or text)
            config = next(connection)

            # Break out if a connection was established but no data was received
            if ((config is None) & (connection.is_exhausted is True)):
                logging.info("Connection closed without any data received")
                return

            # Second messages is the metadata (text)
            metadata_xml = next(connection)
            logging.debug("XML Metadata: %s", metadata_xml)
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if (metadata.acquisitionSystemInformation.systemFieldStrength_T != None):
                    logging.info("Data is from a %s %s at %1.1fT", metadata.acquisitionSystemInformation.systemVendor,
                                 metadata.acquisitionSystemInformation.systemModel,
                                 metadata.acquisitionSystemInformation.systemFieldStrength_T)
            except:
                logging.warning("Metadata is not a valid MRD XML structure.  Passing on metadata as text")
                metadata = metadata_xml

            # Decide what program to use based on config
            # If not one of these explicit cases, try to load file matching name of config
            if config == "simplefft":
                logging.info("Starting simplefft processing based on config")
                simplefft.process(connection, config, metadata)
            if config == "mrd2dicom":
                logging.info("Starting mrd2dicom processing based on config")
                mrd2dicom.process(connection, config, metadata)
            elif config == "invertcontrast":
                logging.info("Starting invertcontrast processing based on config")
                invertcontrast.process(connection, config, metadata)
            elif config == "autoplan_fetal_cardiac_landmarks":
                logging.info("Starting autoplan_fetal_cardiac_landmarks processing based on config")
                autoplan_fetal_cardiac_landmarks.process(connection, config, metadata)
            elif config == "automated_volumetry":
                logging.info("Starting automated_volumetry processing based on config")
                automated_volumetry.process(connection, config, metadata)
            elif config == "autoplan_fetal_sagittal_landmarks":
                logging.info("Starting autoplan_fetal_sagittal_landmarks processing based on config")
                autoplan_fetal_sagittal_landmarks.process(connection, config, metadata)
            elif config == "autoplan_fetal_head_landmarks":
                logging.info("Starting autoplan_fetal_head_landmarks processing based on config")
                autoplan_fetal_head_landmarks.process(connection, config, metadata)
            elif config == "autoplan_fetal_head_landmarks_offline":
                logging.info("Starting autoplan_fetal_head_landmarks_offline processing based on config")
                autoplan_fetal_head_landmarks_offline.process(connection, config, metadata)
            elif config == "autoplan_fetal_head_landmarks_phantom":
                logging.info("Starting autoplan_fetal_head_landmarks_phantom processing based on config")
                autoplan_fetal_head_landmarks_phantom.process(connection, config, metadata)
            elif config == "autoplan_fetal_head_landmarks_phantom_offline":
                logging.info("Starting autoplan_fetal_head_landmarks_phantom_offline processing based on config")
                autoplan_fetal_head_landmarks_phantom_offline.process(connection, config, metadata)
            elif config == "automated_svr_launch":
                logging.info("Starting automated_svr_launch processing based on config")
                automated_svr_launch.process(connection, config, metadata)
            elif config == "automated_pull_svr_brain":
                logging.info("Starting automated_pull_svr_brain processing based on config")
                automated_pull_svr_brain.process(connection, config, metadata)
            elif config == "automated_pull_svr_body":
                logging.info("Starting automated_pull_svr_body processing based on config")
                automated_pull_svr_body.process(connection, config, metadata)
            elif config == "null":
                logging.info("No processing based on config")
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            elif config == "savedataonly":
                # Dummy loop with no processing
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            else:
                try:
                    # Load module from file having exact name as config
                    module = importlib.import_module(config)
                    logging.info("Starting config %s", config)
                    # module.process(connection, config, metadata)
                    invertcontrast.process(connection, config, metadata)

                except ImportError:
                    logging.info("Unknown config '%s'.  Falling back to 'invertcontrast'", config)
                    invertcontrast.process(connection, config, metadata)

        except Exception as e:
            logging.exception(e)

        finally:
            # Encapsulate shutdown in a try block because the socket may have
            # already been closed on the other side
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            logging.info("Socket closed")

            # Dataset may not be closed properly if a close message is not received
            if connection.savedata is True:
                try:
                    connection.dset.close()
                except:
                    pass

                if (connection.savedataFile == ""):
                    try:
                        # Rename the saved file to use the protocol name
                        dset = ismrmrd.Dataset(connection.mrdFilePath, connection.savedataGroup, False)
                        groups = dset.list()

                        if ('xml' in groups):
                            xml_header = dset.read_xml_header()
                            xml_header = xml_header.decode("utf-8")
                            mrdHead = ismrmrd.xsd.CreateFromDocument(xml_header)

                            if (mrdHead.measurementInformation.protocolName != ""):
                                newFilePath = connection.mrdFilePath.replace("MRD_input_", mrdHead.measurementInformation.protocolName + "_")
                                os.rename(connection.mrdFilePath, newFilePath)
                                connection.mrdFilePath = newFilePath
                    except:
                        pass

                if connection.mrdFilePath is not None:
                    logging.info("Incoming data was saved at %s", connection.mrdFilePath)
