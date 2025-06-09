#include <rclcpp/rclcpp.hpp>
#include <rosbag2_interfaces/srv/play_next.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "autoware/control_validator/control_validator.hpp"

#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <nav_msgs/msg/odometry.hpp>

using autoware_planning_msgs::msg::Trajectory;
using nav_msgs::msg::Odometry;
using autoware::control_validator::ControlValidator;
using PlayNextSrv = rosbag2_interfaces::srv::PlayNext;

static rclcpp::NodeOptions create_node_options()
{
    const std::string cv_param =
        ament_index_cpp::get_package_share_directory("autoware_control_validator") +
        "/config/control_validator.param.yaml";
    const std::string vehicle_param =
        ament_index_cpp::get_package_share_directory("autoware_test_utils") +
        "/config/test_vehicle_info.param.yaml";

    rclcpp::NodeOptions opts;
    opts.arguments(std::vector<std::string>{
        "--ros-args", "--params-file", cv_param,
        "--params-file", vehicle_param
    });
    return opts;
}

struct ResultFlags
{
    bool lateral_ok {false};
    bool vel_ok     {false};
};

class CVBagTestNode : public rclcpp::Node
{
public:
    CVBagTestNode(const rclcpp::NodeOptions & opts, ResultFlags & res)
    : Node("cv_bag_test_node", opts), res_(res)
    {
        validator_ = std::make_shared<ControlValidator>(opts);

        pred_sub_ = create_subscription<Trajectory>(
        "/control/trajectory_follower/lateral/predicted_trajectory", 10,
        [this](Trajectory::SharedPtr m){ pred_ = std::move(m); try_eval(); });

        ref_sub_ = create_subscription<Trajectory>(
        "/planning/scenario_planning/lane_driving/trajectory", 10,
        [this](Trajectory::SharedPtr m){ ref_ = std::move(m); try_eval(); });

        odom_sub_ = create_subscription<Odometry>(
        "/localization/kinematic_state", 10,
        [this](Odometry::SharedPtr m){ odom_ = std::move(m); try_eval(); });
    }

    bool isFinished() const { return finished_; }

private:
    void try_eval()
    {
        if (finished_ || !(pred_ && ref_ && odom_)) return;

        validator_->current_predicted_trajectory_ = pred_;
        validator_->current_reference_trajectory_ = ref_;
        validator_->current_kinematics_           = odom_;

        if (!validator_->is_data_ready()) return;

        validator_->validate(*pred_, *ref_, *odom_);
        const auto & st  = validator_->validation_status_;
        const bool   all = validator_->is_all_valid(st);

        res_.lateral_ok = st.is_valid_max_distance_deviation;
        res_.vel_ok     = !st.is_over_velocity;
        finished_       = true;

        RCLCPP_INFO(get_logger(),
        "dist_dev=%.3f (%s) | rolling_back=%s | over_vel=%s  ==> %s",
        st.max_distance_deviation,
        st.is_valid_max_distance_deviation ? "OK" : "NG",
        st.is_rolling_back  ? "YES" : "NO",
        st.is_over_velocity ? "YES" : "NO",
        all ? "PASS" : "FAIL");
    }

    std::shared_ptr<ControlValidator> validator_;
    rclcpp::Subscription<Trajectory>::SharedPtr pred_sub_, ref_sub_;
    rclcpp::Subscription<Odometry>::SharedPtr   odom_sub_;

    Trajectory::SharedPtr pred_, ref_;
    Odometry::SharedPtr   odom_;

    ResultFlags & res_;
    bool finished_{false};
};

int main(int argc, char ** argv)
{
    try {
        rclcpp::init(argc, argv);

        ResultFlags res;
        auto test_node = std::make_shared<CVBagTestNode>(create_node_options(), res);

        auto play_cli = test_node->create_client<PlayNextSrv>("/rosbag2_player/play_next");

        while (!play_cli->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_INFO(test_node->get_logger(), "Waiting for /rosbag2_player/play_next …");
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(test_node->get_logger(), "Interrupted while waiting.");
                rclcpp::shutdown();
                return 1;
            }
        }

        for (int i = 1; i <= 3 && rclcpp::ok(); ++i) {
            RCLCPP_INFO(test_node->get_logger(), ">> play_next() call #%d", i);
            auto fut = play_cli->async_send_request(std::make_shared<PlayNextSrv::Request>());
            rclcpp::spin_until_future_complete(test_node, fut, std::chrono::seconds(15));
            if (test_node->isFinished()) break;
        }

        while (rclcpp::ok() && !test_node->isFinished()) {
            rclcpp::spin_some(test_node);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        const bool ok = res.lateral_ok && res.vel_ok;
        RCLCPP_INFO(test_node->get_logger(), "==== ControlValidator bag test %s ====", ok ? "PASSED" : "FAILED");

        rclcpp::shutdown();
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
