from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from firebase_admin import db, auth
import datetime
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from rest_framework import status

@api_view(['POST'])
@permission_classes([AllowAny]) # Login endpoint itself doesn't require authentication
def admin_login(request):
    """
    Admin login endpoint.
    Authenticates against Django users and returns an API token.
    """
    username = request.data.get('email') # Assuming email is used as username
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'Please provide both email and password'}, status=status.HTTP_400_BAD_REQUEST)

    # Authenticate user using Django's auth system
    # Note: Ensure your AUTHENTICATION_BACKENDS support email/username login if needed
    user = authenticate(request=request, username=username, password=password)

    if user is not None:
        # Check if the user is an admin (staff)
        if user.is_staff:
            # Get or create a token for the user
            token, created = Token.objects.get_or_create(user=user)
            return Response({'token': token.key})
        else:
            # User authenticated but is not an admin
            return Response({'error': 'User is not authorized for admin access'}, status=status.HTTP_403_FORBIDDEN)
    else:
        # Authentication failed
        return Response({'error': 'Invalid Credentials'}, status=status.HTTP_401_UNAUTHORIZED)

@api_view(['POST'])
@permission_classes([IsAdminUser]) # Re-enabled permission check
def admin_logout(request):
    """
    Admin logout endpoint.
    Needs logic to invalidate token/session.
    """
    # For TokenAuthentication, logout is typically handled client-side by deleting the token.
    # If you need server-side token invalidation, you can delete the token:
    try:
        # request.auth is the token object provided by TokenAuthentication
        if request.auth:
            request.auth.delete()
            return Response({'message': 'Logged out successfully by deleting token.'})
        else:
             # This might happen if the request didn't include a valid token
             return Response({'message': 'Logout successful (no token found to delete).'}) 
    except Exception as e:
        print(f"Error during logout: {e}")
        return Response({'error': 'An error occurred during logout.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser]) # Restore IsAdminUser permission check
def admin_user_list_create(request):
    """
    GET: List all users.
    POST: Create a new user.
    Needs implementation using Firebase Admin SDK.
    """
    if request.method == 'GET':
        try:
            users_ref = db.reference('users')
            snapshot = users_ref.get() # Use get() instead of once('value') in Python SDK
            # Firebase returns None if the path doesn't exist or has no data
            if snapshot is None:
                return Response({}) # Return empty dict if no users
            return Response(snapshot)
        except Exception as e:
            print(f"Error fetching users from Firebase: {e}")
            return Response({'error': f'Failed to fetch users: {str(e)}'}, status=500)

    elif request.method == 'POST':
        try:
            email = request.data.get('email')
            password = request.data.get('password')

            if not email or not password:
                return Response({'error': 'Email and password are required'}, status=400)

            # Create the user in Firebase Authentication
            user_record = auth.create_user(
                email=email,
                password=password,
                email_verified=False
            )

            # Create the user in Realtime Database
            user_ref = db.reference(f'users/{user_record.uid}')
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            user_data = {
                'userId': user_record.uid,
                'email': email,
                'accountType': 'free',
                'username': email.split('@')[0],
                'status': 'offline',
                'language': 'English', # Default language
                'translator': 'google', # Default translator
                'createdAt': now,
                'lastLoginDate': now
            }
            user_ref.set(user_data)

            return Response({'message': 'User created successfully', 'userId': user_record.uid}, status=201)
        
        except auth.EmailAlreadyExistsError:
             return Response({'error': 'Email already exists'}, status=400)
        except Exception as e:
            print(f"Error creating user: {e}")
            # Attempt to clean up if user was created in Auth but failed before RTDB
            if 'user_record' in locals() and user_record:
                try:
                    auth.delete_user(user_record.uid)
                    print(f"Cleaned up partially created user: {user_record.uid}")
                except Exception as cleanup_error:
                    print(f"Failed to cleanup partially created user {user_record.uid}: {cleanup_error}")
            return Response({'error': f'Failed to create user: {str(e)}'}, status=500)

@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAdminUser]) # Re-enabled permission check
def admin_user_detail_update_delete(request, user_id):
    """
    GET: Retrieve user details.
    PUT: Update user details.
    DELETE: Delete user.
    Needs implementation using Firebase Admin SDK.
    """
    user_ref = db.reference(f'users/{user_id}')

    if request.method == 'GET':
        try:
            snapshot = user_ref.get()
            if snapshot is None:
                return Response({'error': 'User not found'}, status=404)
            return Response(snapshot)
        except Exception as e:
            print(f"Error fetching user {user_id}: {e}")
            return Response({'error': f'Failed to fetch user: {str(e)}'}, status=500)

    elif request.method == 'PUT':
        try:
            # Check if user exists in RTDB before attempting update
            if user_ref.get() is None:
                 return Response({'error': 'User not found in Realtime Database'}, status=404)

            user_data = request.data
            # Ensure userId is not part of the update payload
            if 'userId' in user_data:
                del user_data['userId'] 
            if 'email' in user_data:
                 # Updating email in RTDB might be okay, but Auth email is separate
                 # Consider if you need to update Auth email as well (requires re-auth typically)
                 # For now, we only update RTDB as per Node.js logic.
                 pass 
            
            # Prevent overwriting critical fields if not provided
            # Ensure essential keys like createdAt are not accidentally removed if not in request.data
            # It might be safer to fetch existing data and merge updates
            # However, mirroring Node.js simple update for now:
            user_ref.update(user_data)
            
            return Response({'message': 'User updated successfully'})
        except Exception as e:
            print(f"Error updating user {user_id}: {e}")
            return Response({'error': f'Failed to update user: {str(e)}'}, status=500)

    elif request.method == 'DELETE':
        try:
            # Check if user exists in RTDB before attempting delete
            if user_ref.get() is None:
                print(f"User {user_id} not found in Realtime Database, attempting Auth delete only.")
                # Proceed to delete from Auth even if not in RTDB
            else:
                # Delete user from Realtime Database first
                 user_ref.delete()
                 print(f"Deleted user {user_id} from Realtime Database.")

            # Delete user from Authentication
            try:
                auth.delete_user(user_id)
                print(f"Deleted user {user_id} from Firebase Authentication.")
            except auth.UserNotFoundError:
                 print(f"User {user_id} not found in Firebase Authentication (already deleted or never existed?).")
                 # If user wasn't in RTDB either, return 404. Otherwise, it's a partial success.
                 if user_ref.get() is None: # Check RTDB again in case of race condition? No, just use initial check. Assume RTDB delete worked if attempted.
                     return Response({'error': 'User not found in RTDB or Auth'}, status=404)
                 else: # User was deleted from RTDB but not found in Auth
                    return Response({'message': 'User deleted from Realtime Database, but not found in Authentication.'}) 

            return Response({'message': 'User deleted successfully'})
        except Exception as e:
            print(f"Error deleting user {user_id}: {e}")
            # Consider potential partial deletion scenarios
            return Response({'error': f'Failed to delete user: {str(e)}'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser]) # Re-enabled permission check
def admin_usage_stats(request):
    """
    Get usage statistics (placeholder).
    Needs implementation using Firebase Admin SDK.
    """
    # TODO: Implement usage stats calculation based on Firebase RTDB data
    try:
        users_ref = db.reference('users')
        users_snapshot = users_ref.get()
        users = users_snapshot if users_snapshot else {}

        daily_login_usage = []
        today = datetime.date.today()

        # Calculate logins for the last 7 days (today + 6 previous days)
        for i in range(6, -1, -1):
            target_date = today - datetime.timedelta(days=i)
            date_str = target_date.isoformat() # YYYY-MM-DD format
            
            users_logged_in_on_date = []
            count = 0
            for user_id, user_data in users.items():
                last_login_iso = user_data.get('lastLoginDate')
                if last_login_iso:
                    try:
                        # Extract date part from ISO string (e.g., "2024-07-28T10:00:00Z")
                        last_login_date_str = last_login_iso.split('T')[0]
                        if last_login_date_str == date_str:
                            count += 1
                            users_logged_in_on_date.append({
                                'userId': user_id, # Include userId for potential UI use
                                'email': user_data.get('email'),
                                'accountType': user_data.get('accountType'),
                                'loginTime': last_login_iso
                            })
                    except Exception as date_parse_error:
                        # Log if a date string is invalid, but continue
                        print(f"Could not parse lastLoginDate '{last_login_iso}' for user {user_id}: {date_parse_error}")
                        continue 

            daily_login_usage.append({
                'date': date_str,
                'count': count,
                'users': users_logged_in_on_date
            })

        return Response({
            'dailyLoginUsage': daily_login_usage
        })

    except Exception as e:
        print(f'Error fetching usage statistics: {e}')
        return Response({'error': 'Failed to fetch usage statistics'}, status=500) 