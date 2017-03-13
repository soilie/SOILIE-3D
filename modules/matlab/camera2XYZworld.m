function [XYZworld, valid] = camera2XYZworld(XYZcamera,extrinsics)
    valid = logical(XYZcamera(:,:,4));
    validXYZ = valid(:);
    XYZ = reshape(XYZcamera,[],4)';
    XYZ = XYZ(1:3,validXYZ);
    XYZworld = transformPointCloud(XYZ,extrinsics);
end