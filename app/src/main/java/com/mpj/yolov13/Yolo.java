/**
 * @author mpj
 * @date 2025/6/23 23:55
 * @version V1.0
 * @since jdk1.8
 **/
package com.mpj.yolov13;

import android.content.res.AssetManager;
import android.view.Surface;

public class Yolo {
	public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);

	public native boolean openCamera(int facing);

	public native boolean closeCamera();

	public native boolean setOutputWindow(Surface surface);

	static {
		System.loadLibrary("yolov13");
	}
}
