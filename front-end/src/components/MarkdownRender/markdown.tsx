"use client";

import { useEffect, useState } from "react";
import { remark } from "remark";
import html from "remark-html";
import styles from "./markdown.module.css";

export function MarkdownRenderer({ markdownText }: { markdownText: string }) {
    const [contentHtml, setContentHtml] = useState<string>("");

    useEffect(() => {
        const processMarkdown = async () => {
            const cleanedMarkdown = markdownText
                .split("\n")
                .filter((line) => line.trim() !== "") // Loại bỏ dòng trống
                .join("\n");

            const processedContent = await remark()
                .use(html)
                .process(cleanedMarkdown);

            // Loại bỏ các thẻ <p> rỗng nếu có
            const sanitizedContent = processedContent
                .toString()
                .replace(/<p>\s*<\/p>/g, "");
            setContentHtml(sanitizedContent);
        };
        processMarkdown();
    }, [markdownText]);

    return (
        <div
            className={styles.prose}
            dangerouslySetInnerHTML={{ __html: contentHtml }}
        />
    );
}

export default MarkdownRenderer;
