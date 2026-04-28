#include <ambf_server/ambf_ral.h>

#if AMBF_ROS1

ambf_ral::ral::ral(int & argc, char * argv[], const std::string & node_name, bool anonymous_name)
{
    init(argc, argv, node_name, anonymous_name);
}

ambf_ral::ral::ral(const std::string & node_name, bool anonymous_name)
{
    // create fake argc/argv
    typedef char * char_pointer;
    char_pointer * argv = new char_pointer[1];
    argv[0]= new char[node_name.size() + 1];
    strcpy(argv[0], node_name.c_str());
    int argc = 1;
    init(argc, argv, node_name, anonymous_name);
}

void ambf_ral::ral::init(int & argc, char * argv[], const std::string & node_name, bool anonymous_name)
{
    if (anonymous_name) {
        ros::init(argc, argv, node_name, ros::init_options::AnonymousName);
    } else {
        ros::init(argc, argv, node_name);
    }
    for (int i = 0; i < argc; ++i) {
        m_stripped_arguments.push_back(argv[i]);
    }
    m_node = std::make_shared<ros::NodeHandle>();
}

ambf_ral::ral::~ral()
{
}

#elif AMBF_ROS2

ambf_ral::ral::ral(int & argc, char * argv[], const std::string & node_name, bool anonymous_name)
{
    init(argc, argv, node_name, anonymous_name);
}

ambf_ral::ral::ral(const std::string & node_name, bool)
{
    const std::string ambf_context = "AMBF";
    if (!rclcpp::ok()) {
        // create fake argc/argv
        typedef char * char_pointer;
        char_pointer * argv = new char_pointer[1];
        argv[0]= new char[ambf_context.size() + 1];
        strcpy(argv[0], ambf_context.c_str());
        int argc = 1;
        rclcpp::init(argc, argv);
    }
    std::string _real_name = node_name;
    clean_nodename(_real_name);
    m_node = std::make_shared<rclcpp::Node>(_real_name);
    m_stripped_arguments.push_back(_real_name);
}

void ambf_ral::ral::init(int & argc, char * argv[], const std::string & node_name, bool)
{
    m_stripped_arguments = rclcpp::init_and_remove_ros_arguments(argc, argv);
    // reconstruct argc/argv from stripped arguments
    typedef char * char_ptr;
    argc = m_stripped_arguments.size();
    argv = reinterpret_cast<char_ptr *>(malloc(argc * sizeof(char_ptr)));
    for (int i = 0; i < argc; ++i) {
        argv[i] = reinterpret_cast<char_ptr>(malloc(m_stripped_arguments.at(i).size() + 1));
        strcpy(argv[i], m_stripped_arguments.at(i).c_str());
    }
    m_node = std::make_shared<rclcpp::Node>(node_name);
}

ambf_ral::ral::~ral()
{
}

#endif
