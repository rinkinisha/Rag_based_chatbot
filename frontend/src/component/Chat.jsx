import { useState } from "react";
import Message from "./Message";

function Chat() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello! Ask me anything about React, Node.js, or MongoDB."
    }
  ]);

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage(event) {
    event.preventDefault();

    if (!question.trim() || loading) {
      return;
    }

    const userQuestion = question.trim();

    // Add user's message immediately
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: userQuestion
      }
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            question: userQuestion,
            history: messages
          })
        }
      );

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();

      // Add assistant response
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources
        }
      ]);
    } catch (error) {
      console.error(error);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content:
            "Sorry, something went wrong while contacting the server."
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-container">

      <div className="chat-header">
        <h1>Knowledge Base RAG</h1>

        <p>
          Ask questions about React, Node.js, and MongoDB
        </p>
      </div>


      <div className="messages">

        {messages.map((message, index) => (
          <Message
            key={index}
            message={message}
          />
        ))}

        {loading && (
          <div className="message assistant">
            Thinking...
          </div>
        )}

      </div>


      <form
        className="input-area"
        onSubmit={sendMessage}
      >

        <input
          type="text"
          value={question}
          onChange={(event) =>
            setQuestion(event.target.value)
          }
          placeholder="Ask a question..."
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading}
        >
          {loading ? "Thinking..." : "Send"}
        </button>

      </form>

    </div>
  );
}

export default Chat;