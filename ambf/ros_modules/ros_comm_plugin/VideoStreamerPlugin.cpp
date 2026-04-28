#include "VideoStreamerPlugin.h"

AF_REGISTER_OBJECT_PLUGIN(afCameraVideoStreamerPlugin);

#ifdef AF_ENABLE_OPEN_CV_SUPPORT

image_transport::ImageTransport* afCameraVideoStreamerPlugin::s_imageTransport = nullptr;
int afCameraVideoStreamerPlugin::init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr a_objectAttribs)
{
    m_objectPtr = a_afObjectPtr;
    m_cameraPtr = (afCameraPtr)a_afObjectPtr;
    afCameraAttributes* camAttribs = (afCameraAttributes*) a_objectAttribs;
    m_rosNode = afROSNode::getNodeAndRegister(m_cameraPtr->getQualifiedName());
    if (s_imageTransport == nullptr) {
#if AMBF_ROS1
        s_imageTransport = new image_transport::ImageTransport(*m_rosNode);
#elif AMBF_ROS2
        s_imageTransport = new image_transport::ImageTransport(m_rosNode);
#endif
    }
    m_imagePublisher = s_imageTransport->advertise(m_cameraPtr->getQualifiedName() + "/ImageData", 1);

    m_publishInterval = camAttribs->m_publishImageInterval;
    return 1;
}

void afCameraVideoStreamerPlugin::graphicsUpdate()
{
    if (m_write_count % m_publishInterval == 0){
        // UGLY HACK TO FLIP ONCES BEFORE PUBLISHING AND THEN AGAIN AFTER TO HAVE CORRECT MAPPING
        // WITH THE COLORED DETPH POINT CLOUD
        m_cameraPtr->m_bufferColorImage->flipHorizontal();
        m_imageMatrix = cv::Mat(m_cameraPtr->m_bufferColorImage->getHeight(), m_cameraPtr->m_bufferColorImage->getWidth(), CV_8UC4, m_cameraPtr->m_bufferColorImage->getData());
        cv::cvtColor(m_imageMatrix, m_imageMatrix, cv::COLOR_RGBA2RGB);
        AMBF_RAL_MSG_PTR(sensor_msgs, Image) rosMsg = cv_bridge::CvImage(AMBF_RAL_MSG(std_msgs, Header)(), "rgb8", m_imageMatrix).toImageMsg();
        rosMsg->header.stamp = ambf_ral::time_from_seconds(m_cameraPtr->getRenderTimeStamp());
        m_imagePublisher.publish(rosMsg);
        m_cameraPtr->m_bufferColorImage->flipHorizontal();
    }
    m_write_count++;
}

void afCameraVideoStreamerPlugin::physicsUpdate(double)
{

}

bool afCameraVideoStreamerPlugin::close()
{
    if (s_imageTransport != nullptr){
        delete s_imageTransport;
        s_imageTransport = nullptr;
    }
    afROSNode::destroyNode(m_cameraPtr->getQualifiedName());
    return true;
}

#endif // AF_ENABLE_OPEN_CV_SUPPORT
