/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Typeface;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.location.LocationListener;
import android.location.LocationManager;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Model;

import java.io.IOException;
import java.util.List;

public class ClassifierActivity
        extends CameraActivity
        implements
        OnImageAvailableListener,
        SensorEventListener {
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    private long lastProcessingTimeMs;
    private Integer sensorOrientation;
    private Classifier classifier;
    private BorderedText borderedText;
    /**
     * Input image size of the model along x axis.
     */
    private int imageSizeX;
    /**
     * Input image size of the model along y axis.
     */
    private int imageSizeY;

    /**
     * 센서 값
     */
//    public float[] accValues = new float[3];
    public float[] gyroValues = new float[3];

    public int sensorChangeCount = 0;
    public int avgCount=0;
    public int prevCount = 0;
    long startTime;

    final private String TURN_RIGHT = "우회전";
    final private String TURN_LEFT = "좌회전";
    final private String START_ACCELERATION = "출발 가속";
    final private String STOP_ACCELERATION = "정지 가속";
    final private String MOVE = "이동";
    final private String STOP = "정지";

    /**
     * gps
     */

    private String gyro_state = "";
    private String acc_state = "";
    private int source = 100;


    private LocationManager lm;
    private LocationListener ll;
//    private double accVector;
    private double gyroVector;
    private double accVectorSum = 0;
    Queue<Double> accVectorSumQueue = new LinkedList<>();

    private double gyroVectorSum = 0;
    Queue<Double> gyroVectorSumQueue = new LinkedList<>();

    private double gyroSum[] = {0, 0, 0};
    Queue<Float> gyroSumQueueX = new LinkedList<>();
    Queue<Float> gyroSumQueueY = new LinkedList<>();
    Queue<Float> gyroSumQueueZ = new LinkedList<>();

//    private double accSum[] = {0, 0, 0};
//    Queue<Float> accSumQueueX = new LinkedList<>();
//    Queue<Float> accSumQueueY = new LinkedList<>();
//    Queue<Float> accSumQueueZ = new LinkedList<>();

//    private float[] initAccValues = new float[3];

    /**
     * 각 상태마다 자원 점수
     */
    private double[] UserState =
            {
                    1,              // safe driving
                    1.35,           // texting - right
                    1.25,           // talking on the phone - right
                    1.35,           // texting - left
                    1.25,           // talking on the phone - left
                    1.4,            // operating the something
                    1.2,            // drinking
                    1.4,            // reaching behind
                    1.5,            // talking to passenger
            };

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        recreateClassifier(getModel(), getDevice(), getNumThreads());
        if (classifier == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    Queue<Classifier.Recognition> oneSecondResults = new LinkedList<>();
    final Classifier.Recognition[] argmaxResult = new Classifier.Recognition[1];    // argmax의 값 저장
    private boolean sensorChanged = false;


    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final int cropSize = Math.min(previewWidth, previewHeight);

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (classifier != null) {

                            final long startTime = SystemClock.uptimeMillis();
                            /**
                             *  sensor onChange()의 횟수에 따라 적용
                             *  센서의 변화가 5번 일어날 때 1초라고 정의
                             *  이때 argmax 값 도출 필요
                             *  results : 이미지의 결과 도출, list로 표현해서 가장 높은 값이 앞으로 표출된다.
                             *
                             *  Classifier.Recognition 의 변수
                             *  id          : String
                             *  title       : String
                             *  confidence  : flaot
                             *
                             *  oneSecondResults: fps의 수에 맞게 조정
                             **/
                            final List<Classifier.Recognition> results =
                                    classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                            LOGGER.v("Detect: %s", results);
                            Log.d("result: ", results + "");
                            Log.d("sensor Change: ", sensorChangeCount + "");

                            Log.d("Frame: ", CameraActivity.frameCount + "");

                            if (frameCount < CameraConnectionFragment.FPS.getUpper()) {
                                oneSecondResults.add(results.get(0));
                                prevCount = sensorChangeCount;
                                CameraActivity.frameCount++;
                            } else {
                                /**
                                 * argmax 도출 해서 도출
                                 * sensorChange 가 일어날때 마다 실행
                                 *
                                 * */
                                oneSecondResults.poll();
                                oneSecondResults.add(results.get(0));
                                Log.d("prev count: ", prevCount + "");
                                Log.d("sensor Change count: ", sensorChangeCount + "");

                                if(sensorChanged){
                                    assert oneSecondResults.size() < 16;
                                    Log.d("Frame Change: ", "change detected");
                                    Log.d("Queue size: ", oneSecondResults.size() + "");
                                    argmaxResult[0] = argmax(oneSecondResults);
                                    sensorChanged = false;
                                }
                            }
                            Log.d("argmax result", Arrays.asList(argmaxResult) + "");

                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            /**
                                             *  argmax 의 결과값은 센서의 onChange 호출 될 때 마다  도출
                                             * */
                                            showResultsInBottomSheet(Arrays.asList(argmaxResult));
                                            showFrameInfo(previewWidth + "x" + previewHeight);
                                            showCropInfo(imageSizeX + "x" + imageSizeY);
                                            showCameraResolution(cropSize + "x" + cropSize);
                                            showRotationInfo(String.valueOf(sensorOrientation));
                                            showInference(lastProcessingTimeMs + "ms");

                                        }
                                    });
                        }
                        readyForNextImage();
                    }
                });
    }

    /**
     * 분류된 결과를 저장하는 큐에서 가장 높은 정확도를 같고있는 값 도출
     * argmax 값을 통해 운전자의 secondary-task 판별
     *
     * @param queue : 이미지 분류 결과를 저장하는 큐, 사이즈는 15로 유지하도록 함
     * @return 분류의 argmax 값
     */
    protected Classifier.Recognition argmax(Queue<Classifier.Recognition> queue) {
        assert queue != null;
        float max = queue.peek().getConfidence();
        Classifier.Recognition result = queue.peek();

        for (int i = 0; i < queue.size(); i++) {
            Classifier.Recognition temp = queue.poll();
            if (temp.getConfidence() > max) {
                max = temp.getConfidence();
                result = temp;
            }
            queue.add(temp);
        }

        return result;
    }


    @Override
    protected void onInferenceConfigurationChanged() {
        if (rgbFrameBitmap == null) {
            // Defer creation until we're getting camera frames.
            return;
        }
        final Device device = getDevice();
        final Model model = getModel();
        final int numThreads = getNumThreads();
        runInBackground(() -> recreateClassifier(model, device, numThreads));
    }

    private void recreateClassifier(Model model, Device device, int numThreads) {
        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }
        if (device == Device.GPU
                && (model == Model.QUANTIZED_MOBILENET
//                || model == Model.QUANTIZED_EFFICIENTNET
        )) {
            LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show();
                    });
            return;
        }
        try {
            LOGGER.d(
                    "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
            classifier = Classifier.create(this, model, device, numThreads);
        } catch (IOException | IllegalArgumentException e) {
            LOGGER.e(e, "Failed to create classifier.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                    });
            return;
        }

        // Updates the input image size.
        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }


    /**
     * callback 함수
     *
     * @param e 센서에 변화가 감지되면 그 순간의 값을 반환 ( 변화량 x )
     *          제곱 평균으로 순간의 벡터값을 구해줍니다.
     *          <p>
     *          accValues[0,1,2] --> 가속도 x,y,z값
     *          gyroValues[0,1,2] --> 각속도 x,y,z값
     */

    @Override
    public void onSensorChanged(SensorEvent e) {

//        if (e.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
//            accValues = e.values;
////            accerelometerData = new String[]{String.valueOf(accValues[0]), String.valueOf(accValues[1]), String.valueOf(accValues[2])};
////            accView.setText("x : "+accValues[0]+" y : "+accValues[1]+" z : "+accValues[3]);
//            accVector = Math.sqrt(
//                    Math.pow(accValues[0] - initAccValues[0], 2)
//                            + Math.pow(accValues[1] - initAccValues[1], 2)
//                            + Math.pow(accValues[2] - initAccValues[2], 2));
//        }
        if (e.sensor.getType() == Sensor.TYPE_GYROSCOPE) {

            gyroValues = e.values;
//            gyroData = new String[]{String.valueOf(gyroValues[0]), String.valueOf(gyroValues[1]), String.valueOf(gyroValues[2])};
//            gyroView.setText("x : "+gyroValues[0]+" y : "+gyroValues[1]+" z : "+gyroValues[3]);

            gyroVector = Math.sqrt(
                    Math.pow(gyroValues[0], 2)
                            + Math.pow(gyroValues[1], 2)
                            + Math.pow(gyroValues[2], 2));
        }


        /**
         * 센서가 변할때 마다
         * UI 변경
         */


        runOnUiThread(new Runnable() {
            @SuppressLint({"DefaultLocale", "SetTextI18n"})
            @Override
            public void run() {
//                accView.setText(String.format("X = %2d  Y = %2d  Z = %2d",
//                        (int)(accValues[0]*10), (int)(accValues[1]*10), (int)(accValues[2]*10)));
//                vectorView.setText(String.format("%.2f",
//                        accVector));
                gyroVectorView.setText(String.format("%.2f",
                        gyroVector));
                gyroView.setText(String.format("X = %2d  Y = %2d  Z = %2d",
                        (int)(gyroValues[0]*10), (int)(gyroValues[1]*10), (int)(gyroValues[2]*10)));
//                initView.setText(String.format("X = %.2f  Y = %.2f  Z = %.2f",
//                        initAccValues[0], initAccValues[1], initAccValues[2]));

                speedStatusView.setText(mySpeed+"km/h");
                sensorChanged = true;
                sensorChangeCount++;

                if (avgCount < 10 ) {
                    /**
                     * Acc, Gryo 벡터 함
                     */
                    avgCount++;
//                    accVectorSum = accVectorSum + accVector;
//                    accVectorSumQueue.add(accVector);

                    gyroVectorSum = gyroVectorSum + gyroVector;
                    gyroVectorSumQueue.add(gyroVector);

                    /**
                     * 자이로 x,y,z 백터 합
                     */
                    gyroSum[0] = gyroSum[0] + gyroValues[0];
                    gyroSumQueueX.add(gyroValues[0]);

                    gyroSum[1] = gyroSum[1] + gyroValues[1];
                    gyroSumQueueY.add(gyroValues[1]);

                    gyroSum[2] = gyroSum[2] + gyroValues[2];
                    gyroSumQueueZ.add(gyroValues[2]);

                    /**
                     * Acc x,y,z 벡터 합
                     */
//                    accSum[0] = accSum[0] + accValues[0];
//                    accSumQueueX.add(accValues[0]);
//
//                    accSum[1] = accSum[1] + accValues[1];
//                    accSumQueueY.add(accValues[1]);
//
//                    accSum[2] = accSum[2] + accValues[2];
//                    accSumQueueZ.add(accValues[2]);

                    /**
                     * gyro 를 활용한 방향성 나타내기
                     */
//                    statusView.setText(gyro_state = getGyroStatus(
//                            gyroVectorSum / avgCount, gyroSum,
//                            initAccValues));

                    Log.d("avgGyro", gyro_state + "");

                    /**
                     * acc 를 활용한 가속성 나타내기
                     */
//                    accStatusView.setText(acc_state = getAccStatus(
//                            accVectorSum / avgCount, accSum,
//                            initAccValues));
                    /**
                     *
                     *
                     * 인지자원
                     *
                     *
                     *
                     * 여기에서 이동하는지 정지하는지에 따른 인지자원 체크
                     */
                } else {
//                    if(classifier.getFrameCount()%30==0){
//                        Log.d("30frame",
//                                "frame : "+classifier.getFrameCount()+"/ time : "+(SystemClock.uptimeMillis()-startTime)+"");
//                        startTime = SystemClock.uptimeMillis();
//                    }
                    /**
                     * 이후 시계열로 알고리즘
                     */


                    gyroVectorSumQueue.add(gyroVector);
                    gyroVectorSum = gyroVectorSum - gyroVectorSumQueue.poll();
                    gyroVectorSum = gyroVectorSum + gyroVector;


//                    accVectorSumQueue.add(accVector);
//                    accVectorSum = accVectorSum - accVectorSumQueue.poll();
//                    accVectorSum = accVectorSum + accVector;
//
//                    /**
//                     * acc x,y,z 초당 평균값
//                     */
//                    accSumQueueX.add(accValues[0]);
//                    accSum[0] += accValues[0];
//                    accSum[0] -= accSumQueueX.poll();
//
//                    accSumQueueY.add(accValues[1]);
//                    accSum[1] += accValues[1];
//                    accSum[1] -= accSumQueueY.poll();
//
//                    accSumQueueZ.add(accValues[2]);
//                    accSum[2] += accValues[2];
//                    accSum[2] -= accSumQueueZ.poll();
                    /**
                     * gyro x,y,z 초당 평균값
                     */
                    gyroSumQueueX.add(gyroValues[0]);
                    gyroSum[0] += gyroValues[0];
                    gyroSum[0] -= gyroSumQueueX.poll();

                    gyroSumQueueY.add(gyroValues[1]);
                    gyroSum[1] += gyroValues[1];
                    gyroSum[1] -= gyroSumQueueY.poll();

                    gyroSumQueueZ.add(gyroValues[2]);
                    gyroSum[2] += gyroValues[2];
                    gyroSum[2] -= gyroSumQueueZ.poll();

//                    accStatusView.setText(acc_state = getAccStatus(
//                            accVectorSum / avgCount, accSum,
//                            initAccValues));
//
//                    statusView.setText(gyro_state = getGyroStatus(
//                            gyroVectorSum / avgCount, gyroSum,
//                            initAccValues));
                    /**
                     * count --> 1초 동안 센서의 변화 횟수 == callback 함수 호출 횟수
                     * avgGyroVector/count
                     * avgAccVector/count
                     * low pass filter
                     */
                    scoreView.setText("인지부담: " +  String.format("%.2f", getScore((float)gyroVectorSum / avgCount, (float)accVectorSum / avgCount)));
                    sensorChangeCount = 0;
                }
            }
        });
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    @Override
    public void onClick(View view) {

        switch (view.getId()) {
            case R.id.init:
//                initAccValues[0] = accValues[0];
//                initAccValues[1] = accValues[1];
//                initAccValues[2] = accValues[2];
                break;
        }
//        Log.d("init", initAccValues[0] + " " + initAccValues[1] + " " + initAccValues[2]);
    }

    /**
     * 기능
     *
     * @param gyroVector    각속도의 벡터값 ( 평균값 현재는 10회 측정당 평균 --> 노이즈 제거용 )
     * @param avgGyro       각속도의 x,y,z 값  ( 변화량 x , 순간 순간의 가속도 값 )
     * @param initAccValues 초기화 버튼을 눌렀을 때, 그 순간의 가속도  x,y,z값 . 중력가속도때문에 0,0,0을 맞춰주기 위함
     *                      현재 휴대폰 상태를 알 수 있음
     * @return 좌회전 우회전을 판단
     */
    public String getGyroStatus(double gyroVector, double[] avgGyro, float[] initAccValues) {
        int threshold = 2;

        double avgGyroX = avgGyro[0] / sensorChangeCount;
        double avgGyroY = avgGyro[1] / sensorChangeCount;
        double avgGyroZ = avgGyro[2] / sensorChangeCount;


        if (gyroVector > threshold) {
            if (initAccValues[0] > 7) {     // 휴대폰을 가로로 눕혔을 때, 화면의 위쪽이 창가를 향할 때,

                if (avgGyroX > threshold / 2) {
                    return TURN_LEFT;
                } else if (avgGyroX < -(threshold / 2)) {
                    return TURN_RIGHT;
                }
                return "가로 기준 회전";
            } else if (initAccValues[0] < -7) {// 휴대폰을 가로로 눕혔을 때, 화면의 아래쪽이 창가를 향할 때,

                if (avgGyroX > threshold / 2) {

                    return TURN_RIGHT;
                } else if (avgGyroX < -(threshold / 2)) {

                    return TURN_LEFT;
                }
                return "가로 기준 회전";
            } else if (initAccValues[1] > 7) {//휴대폰을 세로로 새웠을 때,
                if (avgGyroY > threshold / 2) {

                    return TURN_LEFT;
                } else if (avgGyroY < -(threshold / 2)) {

                    return TURN_RIGHT;
                }

                return "세로 기준 회전";
            }


            return "아무튼 회전중";
        }


        return "회전안함";
    }

    /**
     * 기능
     *
     * @param speedVector   가속도의 벡터값 ( 평균값 현재는 10회 측정당 평균 --> 노이즈 제거용 )
     * @param avgAcc        가속도의 x,y,z 값  ( 변화량 x , 순간 순간의 가속도 값 )
     *                      <p>
     *                      ex) x =0 , y=9.8 , z=0 --> 휴대폰이 수직으로 서있는 값 9.8은 중력 가속도
     * @param initAccValues 초기화 버튼을 눌렀을 때, 그 순간의 가속도  x,y,z값 . 중력가속도때문에 0,0,0을 맞춰주기 위함
     * @return 현재 상태에 따른 가속상태를 표시  1. 출발할때 가속 2. 정지할때 가속
     */
    public String getAccStatus(double speedVector, double[] avgAcc, float[] initAccValues) {

        Log.d("avgSpeed", avgAcc[0] + " " + avgAcc[1] + " " + avgAcc[2]);
        int threshold = 2;


        double avgAccX = avgAcc[0] / sensorChangeCount;
        double avgAccY = avgAcc[1] / sensorChangeCount;
        double avgAccZ = avgAcc[2] / sensorChangeCount;

        if (speedVector > threshold) {

            if (initAccValues[0] > 7) {     // 휴대폰을 가로로 눕혔을 때, 화면의 위쪽이 창가를 향할 때,

                if (avgAccY - initAccValues[1] > threshold / 2 || avgAccZ - initAccValues[2] < -threshold / 2) {
                    return START_ACCELERATION;
                } else if (avgAccY - initAccValues[1] < -threshold / 2 || avgAccZ - initAccValues[2] > threshold / 2) {
                    return STOP_ACCELERATION;
                }


            } else if (initAccValues[0] < -7) {// 휴대폰을 가로로 눕혔을 때, 화면의 아래쪽이 창가를 향할 때,

                if (avgAccY - initAccValues[1] > threshold / 2 || avgAccZ - initAccValues[2] < -threshold / 2) {
                    return STOP_ACCELERATION;
                } else if (avgAccY - initAccValues[1] < -threshold / 2 || avgAccZ - initAccValues[2] > threshold / 2) {
                    return START_ACCELERATION;
                }

                return "가로 기준 가속 중";

            } else if (initAccValues[1] > 7) {//휴대폰을 세로로 새웠을 때,

                if (avgAccX - initAccValues[0] < -threshold / 2 || avgAccZ - initAccValues[2] < -threshold / 2) {
                    return START_ACCELERATION;
                } else if (avgAccX - initAccValues[0] > threshold / 2 || avgAccZ - initAccValues[2] > threshold / 2) {
                    return STOP_ACCELERATION;
                }
                return "세로 기준 화면";
            }

            return "아무튼 가속중";
        }

        if (mySpeed > 0) {
            return MOVE;
        } else {
            return STOP;
        }
    }

    /**
     * 기능
     * 카메라의 프레임당 결과값을 인덱스로 변환
     *
     * @param str mobilenet 에서 나온 결과값 ( string ) 을 받는다.
     * @return 각 라벨의 인덱스 값 0~9까지를 반환
     */
    public int getIndex(String str) {
        switch (str) {
            case "safe driving":
                return 0;
            case "texting - right":
                return 1;
            case "talking on the phone - right":
                return 2;
            case "texting - left":
                return 3;
            case "talking on the phone - left":
                return 4;
            case "operating the something":
                return 5;
            case "drinking":
                return 6;
            case "reaching behind":
                return 7;
            case "talking to passenger":
                return 8;
        }
        return 0;
    }


    /**
     * 기능
     * 인지 부담을 최종적으로 계산해주는 함수
     *
     * @param avgGyroVector  15프레임당 평균적인 각속도 벡터
     * @param avgSpeedVector 15프레임당 평균적인 가속도 벡터
     * @return
     */
    public float getScore(float avgGyroVector, float avgSpeedVector) {
        int index = 0;

        if(argmaxResult[0] != null){
            index = getIndex(argmaxResult[0].getTitle());
        }
        /**
         * 각 A, B, C, D는 프로젝트에서 정의한 수치들
         *  A: 운전자가 얼마나 운전에 능숙한가 (계수: bad 2 ~ good 1)
         *  B: 운전자가 얼마나 인지처리능력을 갖고 있는가 (계수: bad 2 ~ good 1)
         *  C: 운전자가 차량을 어떻게 조작하는가
         *  D: 운전자가 어떤 secondary-task 를 하는가
         *
         *  C: 속도  * ( 가속 + 각속도 + 1 )  --> 각 변수는 정규화 된 값들
         *
         *  [현재 정규화 구간: Min-Max Scaling 적용]
         *  속도: 0.0 ~ 100.0
         *  가속: 0.0 ~ 0.1
         *  회전: 0.0 ~ 0.2\
         *
         *  TODO : 정규화 구간 (민감도) 조정 
         */
        float normalizedAvgGyroVec = avgGyroVector * 50 / 100;
//        float normalizedSpeedVec = 0 / 250 ;
        /** 속도는 더이상 정규화 적용 x, 속도에 따라 인지 부담 수치 변경폭이 넓음 */
//        float normalizedSpeed = (float) mySpeed * ((float) 5/12);

        float A = 1;
        float B = 1;
        float C = (float)mySpeed * (normalizedAvgGyroVec + 1); // 해당 부분은 speed, avgGyroVec, avgSpeedVec 요소가 들어가야함
        float D = (float)UserState[index];

        this.A.setText(A+"");
        this.B.setText(B+"");
        this.C.setText(C+"");
        this.D.setText(D+"");
        this.normalizedAvgGyroVec.setText(normalizedAvgGyroVec+"");
//        this.normalizedSpeedVec.setText(normalizedSpeedVec+"");
//        this.normalizedSpeed.setText(normalizedSpeed+"");

        Log.d("raw data", "avgGyroVector: "+ avgGyroVector + " | avgSpeedVector: " + avgSpeedVector + " | mySpeed: " + mySpeed);    // raw data logging
//        Log.d("normalized data", "normAvgGyroVector: "+ avgGyroVector + " | normAvgSpeedVector: " + avgSpeedVector + " | normSpeed: " + mySpeeDd);
        return A * B * C * D;
    }

    @Override
    public void onStatusChanged(String provider, int status, Bundle extras) {
    }

    @Override
    public void onProviderEnabled(String provider) {
    }

    @Override
    public void onProviderDisabled(String provider) {
    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {
        super.onPointerCaptureChanged(hasCapture);
    }
}
