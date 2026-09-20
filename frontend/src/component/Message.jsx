function Message({ message }) {
  return (
    <div className={`message ${message.role}`}>
      <div className="message-content">
        {message.content}
      </div>

      {message.sources && message.sources.length > 0 && (
        <div className="sources">
          <strong>Sources:</strong>

          {message.sources.map((source, index) => (
            <div key={index}>
              {source.source} — Page {source.page}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Message;