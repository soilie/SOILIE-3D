
function XYZtransform = transformPointCloud(XYZ,Rt)
    XYZtransform = Rt(1:3,1:3) * XYZ;
end