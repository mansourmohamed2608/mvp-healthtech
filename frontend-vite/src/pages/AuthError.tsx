import { Link } from 'react-router-dom';

export default function AuthError() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4">
      <h1 className="text-2xl font-semibold text-red-600">Authentication failed</h1>
      <p className="text-gray-500 text-sm">
        The identity provider returned an error. Please try again.
      </p>
      <Link to="/login" className="text-blue-600 underline text-sm">
        Back to login
      </Link>
    </div>
  );
}
