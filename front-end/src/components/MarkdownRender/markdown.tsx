"use client";
import { useEffect, useState } from "react";
import { remark } from "remark";
import html from "remark-html";
import "./markdown.module.css";
export function MarkdownRenderer({ markdownText }: { markdownText: string }) {
    const [contentHtml, setContentHtml] = useState<string>("");
    useEffect(() => {
        const processMarkdown = async () => {
            // Xóa dòng trống và khoảng trắng thừa trong markdownText
            const cleanedMarkdown = markdownText
                .split("\n")
                .filter((line) => line.trim() !== "") // Loại bỏ dòng trống
                .join("\n"); // Ghép lại thành chuỗi

            const processedContent = await remark()
                .use(html)
                .process(cleanedMarkdown);
            setContentHtml(processedContent.toString());
        };
        processMarkdown();
    }, [markdownText]);

    return (
        <div>
            <style>{`
                .prose p {
                    margin: 0; /* Loại bỏ khoảng cách mặc định giữa các đoạn */
                    line-height: 1.5; /* Điều chỉnh chiều cao dòng nếu cần */
                }
                .prose > *:not(:last-child) {
                    margin-bottom: 0.5em; /* Chỉ thêm khoảng cách nhỏ giữa các phần tử */
                }
            `}</style>
            <div
                className="prose"
                dangerouslySetInnerHTML={{ __html: contentHtml }}
            ></div>
        </div>
    );
}

export default MarkdownRenderer;
