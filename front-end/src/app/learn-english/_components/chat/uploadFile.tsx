import React from "react";
import { Upload, Button, message } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import type { UploadFile } from "antd/es/upload/interface";
import { useChatContext } from "@/libs/context/chat-context";

const ButtonUploadFile: React.FC = () => {
    const { fileList, setFileList } = useChatContext();

    const handleChange = ({
        file,
        fileList,
    }: {
        file: UploadFile;
        fileList: UploadFile[];
    }) => {
        if (file.status === "done") {
            message.success(`${file.name} file uploaded successfully.`);
        } else if (file.status === "error") {
            message.error(`${file.name} file upload failed.`);
        }
        setFileList(fileList);
    };

    return (
        <div className="absolute bottom-0">
            <Upload
                fileList={fileList}
                onChange={handleChange}
                multiple
                customRequest={({ onSuccess }) => {
                    setTimeout(() => {
                        onSuccess?.("ok"); // Simulate successful upload
                    }, 1000);
                }}
                showUploadList={false}
            >
                <Button icon={<UploadOutlined />}></Button>
            </Upload>
        </div>
    );
};

export default ButtonUploadFile;
