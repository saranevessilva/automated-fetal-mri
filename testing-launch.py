


    command = f''''bash /home/auto-proc-svrtk/scripts/auto-brain-055t-reconstruction.sh \
    /tmp/share/debug/{date_path}/dicoms /tmp/share/debug/{date_path}/{date_path}-result 1 4.5 1.0 1 ; \
    chmod 1777 -R /tmp/share/debug/{date_path}/{date_path}-result ; \
    /bin/MIRTK/build/lib/tools/pad-3d /tmp/share/debug/{date_path}/{date_path}-result/reo-SVR-output-brain.nii.gz /home/ref.nii.gz 160 1 ; \
    /bin/MIRTK/build/lib/tools/edit-image /home/ref.nii.gz /home/ref.nii.gz -dx 1 -dy 1 -dz 1 ; \
    /bin/MIRTK/build/lib/tools/transform-image /tmp/share/debug/{date_path}/{date_path}-result/reo-SVR-output-brain.nii.gz \
    /tmp/share/debug/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz -target /home/ref.nii.gz -interp BSpline ; \
    /bin/MIRTK/build/lib/tools/nan /tmp/share/debug/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz 1000000 ; \
    /bin/MIRTK/build/lib/tools/convert-image /tmp/share/debug/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz \
    /tmp/share/debug/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz -short ; \
    chmod 1777 /tmp/share/debug/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz ; \
    suffix=1; \
    while [ -d "/tmp/share/debug/{date_path}-$suffix" ]; do suffix=$((suffix+1)); done; \
    mv /tmp/share/debug/{date_path} /tmp/share/debug/{date_path}-$suffix; \
    mkdir /tmp/share/debug/{date_path}; \
    chmod 1777 /tmp/share/debug/{date_path}; ' '''
